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

    def test_retriggers_when_an_injection_is_immediately_recrossed(self):
        # An accepted injection whose repair is undone WITHIN the same grid interval
        # (a re-stranding delta right after the off-grid trigger) never splits the
        # crossed run in the sampled frame. The arm must re-trigger one full window
        # after the injection — a live operator re-observes persistent crossing and
        # requests again — not terminate `no_trigger` on a still-crossed day.
        import pandas as pd
        recross = pd.concat([
            synthetic_lake_day(),
            pd.DataFrame([dict(origin_time=DAY_OPEN + s(256), sequence_number=9,
                               side_is_bid=True, price=105.0, size=5.0)]),
        ], ignore_index=True)
        from experiments.snapshot_seed import on_demand_reseed_arm
        # 45 s trigger on the 30 s grid: first trig = 210+45 = 255 (off-grid); the
        # 256 s delta re-strands bid 105 before the 270 s sample.
        frame, meta = on_demand_reseed_arm(
            recross, _true_state_provider, grid=GRID, k=2,
            acceptance=self._acceptance(), trigger_after_crossed_s=45.0,
            max_requests=8)
        log = meta["on_demand"]["request_log"]
        # 255: first trigger (repair undone at 256 within the same interval);
        # 300: window restarted at the injection — the production replay defers this
        #      injection by its stricter EVENT-time clock (crossed since the 256 s
        #      delta = 44 s < 45 s at arrival), so crossing persists;
        # 345: next restart — applied, day repaired from the 360 s sample onward.
        assert [r["requested_ts"] for r in log] == [DAY_OPEN + s(255),
                                                    DAY_OPEN + s(300),
                                                    DAY_OPEN + s(345)]
        assert meta["on_demand"]["terminated"] == "no_trigger"
        post = frame[frame["sample_ts"] >= DAY_OPEN + s(360)]
        assert (post["bid_0_price"] < post["ask_0_price"]).all()

    def test_early_stamped_snapshot_is_injected_at_the_trigger_not_before(self):
        # A provider may return state STAMPED at an earlier cadence point (e.g. the
        # last stored grid second). Injecting it at that earlier stamp would repair
        # samples BEFORE the trigger that authorized the request — a retroactive
        # lookahead. The arm must inject at the trigger time; the pre-trigger sample
        # stays crossed exactly as a live causal system would have seen it.
        def early_stamp_provider(requested_ts):
            snap = _true_state_snapshot(requested_ts - s(15))  # stamped 15 s earlier
            return snap, {"vendor": "synthetic", "at_ts": requested_ts}

        # trigger window 45 s is deliberately OFF the 30 s grid: trig = 210+45 = 255 s
        frame, meta = self._run(early_stamp_provider, trigger_after_crossed_s=45.0)
        log = meta["on_demand"]["request_log"]
        assert log and log[0]["requested_ts"] == DAY_OPEN + s(255)
        assert meta["candidates"]["accepted"][0]["ts"] == DAY_OPEN + s(255)
        at_240 = frame[frame["sample_ts"] == DAY_OPEN + s(240)].iloc[0]
        assert at_240["bid_0_price"] >= at_240["ask_0_price"]  # still crossed pre-trigger
        at_270 = frame[frame["sample_ts"] == DAY_OPEN + s(270)].iloc[0]
        assert at_270["bid_0_price"] < at_270["ask_0_price"]   # repaired after trigger


# ----------------------------------------------------------------- arm parity evaluation
class TestEvaluateArmParity:
    def _arm_and_reference(self):
        from experiments.snapshot_seed import (SnapshotAcceptance, seed_lake_replay)
        from recon.coinapi import reconstruct_coinapi_l2_at_samples
        acceptance = SnapshotAcceptance(min_levels_per_side=2, max_age_s=600.0,
                                        tick_scale=100)
        snap = _true_state_snapshot(DAY_OPEN + s(300))
        arm_frame, arm_meta = seed_lake_replay(
            synthetic_lake_day(), [(snap, {})], grid=GRID, k=2, acceptance=acceptance)
        ref_frame, _ = reconstruct_coinapi_l2_at_samples(
            synthetic_l3_day(), k=2, day=DAY, sample_ts=GRID, size_policy="decrement")
        return arm_frame, arm_meta, ref_frame

    def test_report_structure_and_exclusions(self):
        from experiments.snapshot_seed import evaluate_arm_parity
        arm_frame, arm_meta, ref = self._arm_and_reference()
        rep = evaluate_arm_parity(arm_frame, arm_meta, ref, k=2, grid_s=30.0,
                                  injection_guard_s=60.0)
        assert set(rep) >= {"day_quality", "parity", "parity_guarded", "since_ts",
                            "excluded_crossed_ts", "injection_guard"}
        # A MID-DAY seed (here 300 s in) does NOT clamp the compared window: the
        # pre-seed established cold-start book is the strategy's genuine output and
        # must be scored (documented deviation from run_parity_core, which only ever
        # clamps day-open bootstrap seeds). The warm-up cutoff still applies.
        assert rep["since_ts"] == DAY_OPEN + s(90)  # 3rd consecutive valid sample
        assert rep["parity"]["n_grid_full"] == len(GRID)
        dq = rep["day_quality"]
        assert {"crossed_rate", "missing_book_fraction", "thin_depth_fraction",
                "crossed_duration_s", "seed_accepted"} <= set(dq)
        # guarded variant additionally masks the 60 s after the injection at 300s
        g = rep["injection_guard"]
        assert g["guard_s"] == 60.0
        assert g["n_guard_excluded"] >= 1  # the 300s and 330s samples fall in the guard

    def test_pre_seed_crossed_samples_stay_scored_for_mid_day_seeds(self):
        # On-demand-style arm: the first accepted snapshot lands MID-DAY (300 s), and
        # every crossed sample precedes it (cold-start output at 210..270 s). Those
        # samples are observable strategy behavior: they must NOT enter the
        # crossed-exclusion set, which exists only for residual crossings AWAITING a
        # reseed under an active repair regime (ts >= seed_ts).
        from experiments.snapshot_seed import (SnapshotAcceptance, evaluate_arm_parity,
                                               seed_lake_replay)
        from recon.coinapi import reconstruct_coinapi_l2_at_samples
        acceptance = SnapshotAcceptance(min_levels_per_side=2, max_age_s=600.0,
                                        tick_scale=100)
        snap = _true_state_snapshot(DAY_OPEN + s(300))
        arm_frame, arm_meta = seed_lake_replay(
            synthetic_lake_day(), [(snap, {})], grid=GRID, k=2, acceptance=acceptance)
        assert arm_meta["crossed_sample_ts"]  # crossed samples exist, all pre-seed
        ref, _ = reconstruct_coinapi_l2_at_samples(
            synthetic_l3_day(), k=2, day=DAY, sample_ts=GRID, size_policy="decrement")
        rep = evaluate_arm_parity(arm_frame, arm_meta, ref, k=2, grid_s=30.0)
        assert rep["n_excluded_crossed"] == 0
        assert rep["parity"]["crossed_rate"]["lake"] > 0  # scored, not hidden

    def test_day_open_seed_still_clamps_to_seed_ts(self):
        from experiments.snapshot_seed import (SnapshotAcceptance, evaluate_arm_parity,
                                               seed_lake_replay)
        from recon.coinapi import reconstruct_coinapi_l2_at_samples
        acceptance = SnapshotAcceptance(min_levels_per_side=2, max_age_s=600.0,
                                        tick_scale=100)
        seed = _true_state_snapshot(DAY_OPEN)  # a true day-open bootstrap
        arm_frame, arm_meta = seed_lake_replay(
            synthetic_lake_day(), [(seed, {})], grid=GRID, k=2, acceptance=acceptance)
        ref, _ = reconstruct_coinapi_l2_at_samples(
            synthetic_l3_day(), k=2, day=DAY, sample_ts=GRID, size_policy="decrement")
        rep = evaluate_arm_parity(arm_frame, arm_meta, ref, k=2, grid_s=30.0)
        # production clamp semantics: max(warm-up cutoff, seed_ts); warm-up needs 3
        # consecutive good samples, so the cutoff is the 60 s sample
        assert rep["since_ts"] == DAY_OPEN + s(60)

    def test_parity_carries_required_metrics(self):
        from experiments.snapshot_seed import evaluate_arm_parity
        arm_frame, arm_meta, ref = self._arm_and_reference()
        rep = evaluate_arm_parity(arm_frame, arm_meta, ref, k=2, grid_s=30.0)
        p = rep["parity"]
        assert {"mid_diff", "label_agreement", "spike_counts",
                "crossed_rate", "missing_book"} <= set(p)
        assert {"median", "p95", "p99", "corr"} <= set(p["mid_diff"])
        assert set(p["label_agreement"]) == {"2", "10", "60"}

    def test_deterministic_report(self):
        from experiments.snapshot_seed import evaluate_arm_parity
        arm_frame, arm_meta, ref = self._arm_and_reference()
        r1 = evaluate_arm_parity(arm_frame, arm_meta, ref, k=2, grid_s=30.0)
        r2 = evaluate_arm_parity(arm_frame, arm_meta, ref, k=2, grid_s=30.0)
        assert r1["report_hash"] == r2["report_hash"]


# --------------------------------------------------------------- preregistered thresholds
class TestPreregistration:
    def test_json_artifact_matches_module_constants(self):
        import json
        import pathlib
        from experiments.snapshot_seed import PREREGISTERED
        artifact = json.loads(
            (pathlib.Path(__file__).parent.parent / "experiments" /
             "preregistration_54.json").read_text())
        assert artifact["thresholds"] == PREREGISTERED["thresholds"]
        assert artifact["fixture_days"] == PREREGISTERED["fixture_days"]

    def _dq(self, **kw):
        d = {"crossed_rate": 0.0001, "missing_book_fraction": 0.001,
             "thin_depth_fraction": 0.01, "crossed_duration_s": 10.0,
             "seed_accepted": True}
        d.update(kw)
        return d

    def _parity(self, **kw):
        p = {"mid_diff": {"median": 0.0, "signed_mean": 0.05, "corr": 0.99999,
                          "p95": 0.5, "p99": 4.0},
             "spike_fraction": {">50": 0.00002},
             "label_agreement": {"2": {"agreement": 0.95},
                                 "10": {"agreement": 0.98},
                                 "60": {"agreement": 0.995}}}
        p.update(kw)
        return p

    def test_passing_metrics_pass(self):
        from experiments.snapshot_seed import evaluate_preregistered
        verdict = evaluate_preregistered(day_quality=self._dq(), parity=self._parity())
        assert verdict["pass"] is True
        assert verdict["failed"] == []

    def test_failing_metrics_name_the_criterion(self):
        from experiments.snapshot_seed import evaluate_preregistered
        verdict = evaluate_preregistered(
            day_quality=self._dq(crossed_rate=0.05),
            parity=self._parity(mid_diff={"median": 0.0, "signed_mean": 0.05,
                                          "corr": 0.9990, "p95": 0.5, "p99": 4.0}))
        assert verdict["pass"] is False
        assert "day_quality.crossed_rate" in verdict["failed"]
        assert "parity.mid_corr" in verdict["failed"]

    def test_unseeded_arm_fails_crossed_duration_closed(self):
        from experiments.snapshot_seed import evaluate_preregistered
        # crossed_duration_s only accumulates after a seed lands (recon.reseed
        # update_crossed), so a never-seeded arm reporting 0.0 has NOT measured the
        # bar — fail closed rather than pass on an unmeasured metric.
        verdict = evaluate_preregistered(
            day_quality=self._dq(seed_accepted=False, crossed_duration_s=0.0),
            parity=self._parity())
        assert "day_quality.crossed_duration_s" in verdict["failed"]

    def test_prereg_arm_names_are_runnable(self):
        import json
        import pathlib
        import importlib
        runner = importlib.import_module("scripts.run_snapshot_seed_experiment")
        artifact = json.loads(
            (pathlib.Path(__file__).parent.parent / "experiments" /
             "preregistration_54.json").read_text())
        specs = runner.arm_specs_from_names(artifact["arms"])  # must not raise
        assert len(specs) == len(artifact["arms"])
        assert artifact["emulated_variants"]  # documented separately from arms


# --------------------------------------------------------------------- economics verdict
class TestEvaluateEconomics:
    def test_passing_band_passes(self):
        from experiments.snapshot_seed import evaluate_economics, project_strategy_costs
        costs = project_strategy_costs(full_day_book_gb=2.0, on_demand_requests=1)
        v = evaluate_economics(costs=costs)
        assert v["pass"] is True and v["failed"] == []
        assert "economics.rest_on_demand" in v["checked"]

    def test_band_worse_than_bar_fails_conservatively(self):
        from experiments.snapshot_seed import evaluate_economics, project_strategy_costs
        # 200 requests at the 10-credit cap on a small day: the HIGH cost end busts
        # the <=25%-of-full-day bar even though the LOW end passes -> fail (bands are
        # resolved conservatively, never optimistically).
        costs = project_strategy_costs(full_day_book_gb=0.8, on_demand_requests=200)
        v = evaluate_economics(costs=costs)
        assert v["pass"] is False
        assert "economics.rest_on_demand" in v["failed"]

    def test_missing_band_fails_closed(self):
        from experiments.snapshot_seed import evaluate_economics
        v = evaluate_economics(costs={"full_day_fill": {"usd": 2.0}})
        assert v["pass"] is False and v["failed"] == ["economics.no_strategy_band"]


# ------------------------------------------------------------------------ cost projection
class TestCostProjection:
    def test_full_day_baseline_and_strategy_bands(self):
        from experiments.snapshot_seed import project_strategy_costs
        costs = project_strategy_costs(
            full_day_book_gb=2.266,
            on_demand_requests=3,
            stream_stats={"n_changed": 40_000, "max_levels": 20},
        )
        base = costs["full_day_fill"]
        assert base["usd"] == pytest.approx(2.266 * 1.0 + 0.01)  # $1/GB + 1 GET
        od = costs["rest_on_demand"]
        assert od["n_requests"] == 3
        # band: [1 credit/request, levels-as-data-items worst case], first-1k pricing
        assert od["usd_band"]["low"] == pytest.approx(3 * 1 * 5.26 / 1000)
        assert od["usd_band"]["high"] > od["usd_band"]["low"]
        st = costs["flatfile_snapshot_stream"]
        assert st["rows"] == 40_000
        assert 0 < st["usd_band"]["low"] < st["usd_band"]["high"]
        assert costs["billing_facts"]["book_usd_per_gb"] == 1.0

    def test_savings_versus_full_day(self):
        from experiments.snapshot_seed import project_strategy_costs
        costs = project_strategy_costs(full_day_book_gb=2.0, on_demand_requests=1,
                                       stream_stats=None)
        od = costs["rest_on_demand"]
        assert od["saving_vs_full_day"]["low"] > 0.9  # >=90% cheaper even at band high


# ------------------------------------------------------------------ degraded-day fixtures
class TestEmulateDegradation:
    def test_leading_gap_drops_rows_before_start(self):
        from experiments.snapshot_seed import emulate_degradation
        out, info = emulate_degradation(synthetic_lake_day(), "leading_gap",
                                        start_ts=DAY_OPEN + s(150))
        assert (out["origin_time"] >= DAY_OPEN + s(150)).all()
        assert info["kind"] == "leading_gap"
        assert info["rows_before"] == 8 and info["rows_after"] == 3

    def test_sparse_drops_deterministic_fraction(self):
        from experiments.snapshot_seed import emulate_degradation
        out1, info1 = emulate_degradation(synthetic_lake_day(), "sparse", keep_mod=2)
        out2, _ = emulate_degradation(synthetic_lake_day(), "sparse", keep_mod=2)
        pd.testing.assert_frame_equal(out1, out2)  # deterministic, no RNG
        assert 0 < len(out1) < 8
        assert info1["rows_after"] == len(out1)

    def test_input_frame_is_not_mutated(self):
        from experiments.snapshot_seed import emulate_degradation
        df = synthetic_lake_day()
        before = df.copy()
        emulate_degradation(df, "leading_gap", start_ts=DAY_OPEN + s(150))
        pd.testing.assert_frame_equal(df, before)

    def test_unknown_kind_fails_loudly(self):
        from experiments.snapshot_seed import emulate_degradation
        with pytest.raises(ValueError, match="unknown degradation"):
            emulate_degradation(synthetic_lake_day(), "bogus")


# --------------------------------------------------------- clean-control non-regression
class TestControlNonRegression:
    def _metrics(self, crossed=0.0001, p99=4.0, corr=0.99999, label2=0.95):
        return {"day_quality": {"crossed_rate": crossed},
                "parity": {"mid_diff": {"p99": p99, "corr": corr},
                           "label_agreement": {"2": {"agreement": label2}}}}

    def test_matching_control_passes(self):
        from experiments.snapshot_seed import evaluate_control_non_regression
        v = evaluate_control_non_regression(arm=self._metrics(),
                                            control=self._metrics())
        assert v["pass"] is True and v["failed"] == []

    def test_regression_names_the_criterion(self):
        from experiments.snapshot_seed import evaluate_control_non_regression
        v = evaluate_control_non_regression(
            arm=self._metrics(crossed=0.01, p99=8.0, corr=0.9990, label2=0.90),
            control=self._metrics())
        assert v["pass"] is False
        assert set(v["failed"]) == {"non_regression.crossed_rate",
                                    "non_regression.mid_p99",
                                    "non_regression.mid_corr",
                                    "non_regression.label_2s"}


# ---------------------------------------------------------------- cached-day Lake loader
class TestLoadLakeCachedDay:
    def _make_cache(self, tmp_path, url, df):
        import hashlib as _h
        import json
        import joblib
        d = (tmp_path / "joblib" / "lakeapi" / "main" / "_download_one" /
             _h.md5(url.encode()).hexdigest())
        d.mkdir(parents=True)
        (d / "metadata.json").write_text(
            json.dumps({"duration": 1.0, "input_args": {"url": f"'{url}'"}}))
        joblib.dump(df, d / "output.pkl")

    def test_loads_and_concats_matching_day_files(self, tmp_path):
        from experiments.snapshot_seed import load_lake_cached_day
        base = ("https://data.crypto-lake.com/market-data/cryptofeed/book_delta_v2/"
                "exchange=COINBASE/symbol=BTC-USD/dt=2025-06-01/")
        df1 = pd.DataFrame({"timestamp": [1, 2], "sequence_number": [1, 2],
                            "side_is_bid": [True, False], "price": [1.0, 2.0],
                            "size": [1.0, 1.0]})
        df2 = pd.DataFrame({"timestamp": [3], "sequence_number": [3],
                            "side_is_bid": [True], "price": [3.0], "size": [1.0]})
        self._make_cache(tmp_path, base + "2.snappy.parquet", df2)
        self._make_cache(tmp_path, base + "1.snappy.parquet", df1)
        out, info = load_lake_cached_day(tmp_path, table="book_delta_v2",
                                         exchange="COINBASE", symbol="BTC-USD",
                                         day="2025-06-01")
        assert list(out["timestamp"]) == [1, 2, 3]  # file order by url
        assert info["n_files"] == 2 and len(info["files"]) == 2

    def test_missing_day_fails_loudly(self, tmp_path):
        from experiments.snapshot_seed import load_lake_cached_day
        (tmp_path / "joblib" / "lakeapi" / "main" / "_download_one").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="2025-06-01"):
            load_lake_cached_day(tmp_path, table="book_delta_v2", exchange="COINBASE",
                                 symbol="BTC-USD", day="2025-06-01")

    def test_matched_shard_with_missing_body_fails_instead_of_partial_load(self, tmp_path):
        # A matched metadata.json whose output.pkl is missing (interrupted/evicted
        # cache) must FAIL the load — silently proceeding on the remaining shards
        # would replay a partial Lake day and corrupt parity while reporting normally.
        import json
        from experiments.snapshot_seed import load_lake_cached_day
        base = ("https://data.crypto-lake.com/market-data/cryptofeed/book_delta_v2/"
                "exchange=COINBASE/symbol=BTC-USD/dt=2025-06-01/")
        df = pd.DataFrame({"timestamp": [1], "sequence_number": [1],
                           "side_is_bid": [True], "price": [1.0], "size": [1.0]})
        self._make_cache(tmp_path, base + "1.snappy.parquet", df)
        # second shard: metadata only, body missing
        import hashlib as _h
        d = (tmp_path / "joblib" / "lakeapi" / "main" / "_download_one" /
             _h.md5((base + "2.snappy.parquet").encode()).hexdigest())
        d.mkdir(parents=True)
        (d / "metadata.json").write_text(
            json.dumps({"input_args": {"url": f"'{base}2.snappy.parquet'"}}))
        with pytest.raises(FileNotFoundError, match="2.snappy.parquet"):
            load_lake_cached_day(tmp_path, table="book_delta_v2", exchange="COINBASE",
                                 symbol="BTC-USD", day="2025-06-01")

    def test_evidence_index_commits_are_resolvable_shas(self):
        import json
        import pathlib
        import re
        idx = json.loads((pathlib.Path(__file__).parent.parent / "experiments" /
                          "evidence_54" / "index_54.json").read_text())
        # provenance fields must be bare 40-hex SHAs a reviewer can `git rev-parse`
        for field in ("reports_generated_at_commit", "evidence_rebuilt_at_commit"):
            assert re.fullmatch(r"[0-9a-f]{40}", idx[field]), field


# ------------------------------------------------------------------- day orchestration
class TestRunExperimentDay:
    def _run(self, **kw):
        from experiments.snapshot_seed import SnapshotAcceptance, run_experiment_day
        from recon.coinapi import reconstruct_coinapi_l2_at_samples
        ref, _ = reconstruct_coinapi_l2_at_samples(
            synthetic_l3_day(), k=2, day=DAY, sample_ts=GRID, size_policy="decrement")
        defaults = dict(
            day=DAY, lake_df=synthetic_lake_day(),
            coinapi_chunks_factory=lambda: synthetic_l3_day(),
            reference_frame=ref, grid=GRID, k=2,
            acceptance=SnapshotAcceptance(min_levels_per_side=2, max_age_s=600.0,
                                          tick_scale=100),
            arm_specs=[{"name": "cold_control", "kind": "cold"},
                       {"name": "lake_book_control", "kind": "lake_book"},
                       {"name": "coinapi_day_open_L2", "kind": "day_open", "levels": 2},
                       {"name": "coinapi_stream_L2", "kind": "stream", "levels": 2},
                       {"name": "coinapi_on_demand_L2", "kind": "on_demand",
                        "levels": 2}],
            lake_book_snapshots=[_true_state_snapshot(DAY_OPEN + s(60))],
            trigger_after_crossed_s=30.0,
            full_day_book_gb=2.0,
            input_info={"coinapi_parquet_sha256": "deadbeef"},
        )
        defaults.update(kw)
        return run_experiment_day(**defaults)

    def test_report_covers_all_arms_with_evaluations(self):
        rep = self._run()
        assert set(rep["arms"]) == {"cold_control", "lake_book_control",
                                    "coinapi_day_open_L2", "coinapi_stream_L2",
                                    "coinapi_on_demand_L2"}
        for name, arm in rep["arms"].items():
            assert "evaluation" in arm and "meta" in arm, name
            assert arm["meta"]["frame_hash"], name
            assert "preregistered" in arm["evaluation"], name
        assert rep["inputs"]["coinapi_parquet_sha256"] == "deadbeef"
        assert rep["preregistration"]["thresholds"] == __import__(
            "experiments.snapshot_seed", fromlist=["PREREGISTERED"]
        ).PREREGISTERED["thresholds"]

    def test_on_demand_arm_carries_request_log_and_costs(self):
        rep = self._run()
        od = rep["arms"]["coinapi_on_demand_L2"]
        assert od["meta"]["on_demand"]["n_requests"] >= 1
        assert "rest_on_demand" in od["costs"]
        st = rep["arms"]["coinapi_stream_L2"]
        assert "flatfile_snapshot_stream" in st["costs"]
        assert st["costs"]["flatfile_snapshot_stream"]["rows"] > 0

    def test_non_regression_computed_against_lake_book_control(self):
        rep = self._run()
        for name in ("coinapi_day_open_L2", "coinapi_stream_L2"):
            assert rep["arms"][name]["non_regression"] is not None

    def test_day_open_arm_is_seed_only_and_keeps_residual_crossed(self):
        rep = self._run()
        arm = rep["arms"]["coinapi_day_open_L2"]
        # production --no-reseed A/B semantics: a single day-open snapshot with no
        # intraday repair; residual crossed samples surface in parity, never excluded
        assert arm["meta"]["policy"]["enabled"] is False
        assert arm["evaluation"]["n_excluded_crossed"] == 0

    def test_guarded_verdict_scored_and_gates_shared_source_arms(self):
        rep = self._run()
        for name, arm in rep["arms"].items():
            ev = arm["evaluation"]
            assert "preregistered_guarded" in ev, name
            kind = arm["spec"]["kind"]
            if kind in ("day_open", "stream", "on_demand"):
                assert ev["preregistered_guarded"] is not None, name
                econ = arm["economics"]
                nr = arm["non_regression"]
                expected = bool(ev["preregistered"]["pass"]
                                and ev["preregistered_guarded"]["pass"]
                                and econ is not None and econ["pass"]
                                and (nr is None or nr["pass"]))
                assert arm["prereg_pass_effective"] == expected, name
                assert nr is not None, name  # control arm present in this run
            else:
                assert arm["prereg_pass_effective"] == ev["preregistered"]["pass"], name

    def test_effective_pass_gates_on_economics_for_priced_arms(self):
        # The preregistered GO gate includes the <=25%-of-full-day economics bar:
        # a snapshot arm that passes parity + guarded parity but is uneconomic (or
        # has no computable band — fail-closed) must not be an effective pass.
        from experiments.snapshot_seed import effective_prereg_pass
        ok = {"pass": True}
        bad = {"pass": False}
        assert effective_prereg_pass("stream", ok, ok, ok) is True
        assert effective_prereg_pass("stream", ok, ok, bad) is False
        assert effective_prereg_pass("on_demand", ok, ok, None) is False  # fail-closed
        assert effective_prereg_pass("day_open", ok, bad, ok) is False
        assert effective_prereg_pass("stream", bad, ok, ok) is False
        # the preregistered clean-control non-regression gate feeds the verdict when
        # a control arm exists; absent control (partial-arm run) it cannot gate
        assert effective_prereg_pass("stream", ok, ok, ok, non_regression=bad) is False
        assert effective_prereg_pass("stream", ok, ok, ok, non_regression=ok) is True
        assert effective_prereg_pass("stream", ok, ok, ok, non_regression=None) is True
        # controls: no snapshot injected, no cost — plain verdict only
        assert effective_prereg_pass("cold", ok, None, None) is True
        assert effective_prereg_pass("lake_book", bad, None, None) is False
        # fail-closed integration: priced arms without a cost band never pass
        rep = self._run(full_day_book_gb=None)
        for name in ("coinapi_day_open_L2", "coinapi_stream_L2",
                     "coinapi_on_demand_L2"):
            arm = rep["arms"][name]
            assert arm["economics"] is None, name
            assert arm["prereg_pass_effective"] is False, name

    def test_stream_arm_meta_hash_covers_stream_stats(self):
        from eval.hashing import hash_obj
        rep = self._run()
        meta = rep["arms"]["coinapi_stream_L2"]["meta"]
        assert "stream_stats" in meta
        # the advertised meta hash must cover EVERYTHING in the meta (a corrupt
        # stream sizing basis must change the hash); note the report meta is the
        # _slim_meta view, whose hash is recomputed after slimming + stats
        assert meta["report_hash"] == hash_obj(meta, exclude_keys=("report_hash",))

    def test_economics_verdict_attached_to_priced_arms(self):
        rep = self._run()
        assert rep["arms"]["coinapi_on_demand_L2"]["economics"] is not None
        assert rep["arms"]["coinapi_stream_L2"]["economics"] is not None
        assert rep["arms"]["cold_control"]["economics"] is None

    def test_unpurchasable_snapshot_depths_are_unpriced_and_fail_closed(self):
        # No documented product sells a FULL-DEPTH (or >20-level) historical snapshot
        # below a full-day file — REST history is hard-capped at 20 levels. Pricing
        # such an arm as a cheap REST request would let an impossible strategy pass
        # the economics gate; it must stay unpriced -> economics None -> effective
        # fail-closed.
        from recon.coinapi import reconstruct_coinapi_l2_at_samples
        deep_ref, _ = reconstruct_coinapi_l2_at_samples(
            synthetic_l3_day(), k=30, day=DAY, sample_ts=GRID, size_policy="decrement")
        rep = self._run(reference_frame=deep_ref, arm_specs=[
            {"name": "coinapi_day_open_full", "kind": "day_open", "levels": None},
            {"name": "coinapi_day_open_L2", "kind": "day_open", "levels": 2},
            {"name": "coinapi_on_demand_L30", "kind": "on_demand", "levels": 30},
        ])
        full = rep["arms"]["coinapi_day_open_full"]
        assert full["costs"] is None and full["economics"] is None
        assert full["prereg_pass_effective"] is False
        deep = rep["arms"]["coinapi_on_demand_L30"]
        assert deep["costs"] is None and deep["economics"] is None
        assert deep["prereg_pass_effective"] is False
        assert rep["arms"]["coinapi_day_open_L2"]["costs"] is not None

    def test_report_is_deterministic_and_json_safe(self):
        import json
        r1 = self._run()
        r2 = self._run()
        assert r1["report_hash"] == r2["report_hash"]
        json.dumps(r1, allow_nan=False)  # strict JSON, no NaN leakage

    def test_large_sample_lists_are_capped_in_report(self):
        rep = self._run()
        for arm in rep["arms"].values():
            meta = arm["meta"]
            assert "crossed_sample_ts" not in meta
            assert len(meta.get("reseed_ts", [])) <= 100
            # candidate ledgers are unbounded on real days (86,400 stream entries);
            # the shipped meta caps them while keeping exact counts and the
            # full-ledger hash (full_meta_hash)
            led = meta["candidates"]
            assert len(led["accepted"]) <= 50 and len(led["rejected"]) <= 50
            assert led["n_total"] >= led["n_accepted"] >= 0
            assert meta["full_meta_hash"]

    def test_slim_meta_caps_oversized_candidate_ledgers(self):
        # Real stream arms carry 86,400 accepted candidates; the shipped meta caps
        # the ledgers (keeping counts + the full-ledger hash) so a day report stays
        # reviewable instead of ballooning to >100 MB.
        from experiments.snapshot_seed import _slim_meta
        meta = {"report_hash": "x", "candidates": {
            "n_total": 200, "n_accepted": 150,
            "accepted": [{"ts": i} for i in range(150)],
            "rejected": [{"ts": i, "reason": "crossed"} for i in range(50, 100)]}}
        slim = _slim_meta(meta)
        assert len(slim["candidates"]["accepted"]) == 50
        assert len(slim["candidates"]["rejected"]) == 50
        assert slim["candidates"]["n_accepted"] == 150  # counts preserved
        assert meta["candidates"]["accepted"][0] == {"ts": 0}  # input untouched
        assert slim["full_meta_hash"] == "x"


# ------------------------------------------------------------- frame snapshot provider
class TestFrameSnapshotProvider:
    def test_returns_state_at_last_sample_at_or_before_request(self):
        from experiments.snapshot_seed import frame_snapshot_provider
        provider = frame_snapshot_provider(_topk_frame(), max_levels=2)
        snap, prov = provider(DAY_OPEN + int(2.5 * NS))
        assert snap.ts == DAY_OPEN + s(2)  # last grid second <= request; causal
        assert snap.asks == ((101.0, 0.5), (102.0, 2.5))
        assert prov["at_ts"] == DAY_OPEN + int(2.5 * NS)
        assert prov["vendor"] == "coinapi"

    def test_truncates_to_max_levels(self):
        from experiments.snapshot_seed import frame_snapshot_provider
        provider = frame_snapshot_provider(_topk_frame(), max_levels=1)
        snap, _ = provider(DAY_OPEN + s(1))
        assert snap.bids == ((100.0, 1.0),) and snap.asks == ((101.0, 1.5),)

    def test_request_before_first_sample_yields_empty_candidate(self):
        from experiments.snapshot_seed import frame_snapshot_provider
        provider = frame_snapshot_provider(_topk_frame(), max_levels=2)
        snap, _ = provider(DAY_OPEN - 1)
        assert snap.bids == () and snap.asks == ()  # classify -> one_sided, rejected


# ----------------------------------------------------------------------------- CLI shell
class TestRunnerScript:
    def test_k_ref_covers_every_reference_backed_arm(self):
        import importlib
        runner = importlib.import_module("scripts.run_snapshot_seed_experiment")
        specs = runner.arm_specs_from_names(
            ["cold_control", "coinapi_on_demand_L20", "coinapi_stream_L5"])
        # on_demand reads the reference frame too — omitting it would crash the
        # frame provider on a k=10 reference
        assert runner.k_ref_for(specs, k=10) == 20
        assert runner.k_ref_for(runner.arm_specs_from_names(["cold_control"]), k=10) == 10

    def test_arm_specs_from_names(self):
        import importlib
        runner = importlib.import_module("scripts.run_snapshot_seed_experiment")
        specs = runner.arm_specs_from_names(
            ["cold_control", "lake_book_control", "coinapi_day_open_L20",
             "coinapi_day_open_full", "coinapi_stream_L5", "coinapi_on_demand_L20"])
        by_name = {sp["name"]: sp for sp in specs}
        assert by_name["cold_control"]["kind"] == "cold"
        assert by_name["lake_book_control"]["kind"] == "lake_book"
        assert by_name["coinapi_day_open_L20"] == {
            "name": "coinapi_day_open_L20", "kind": "day_open", "levels": 20}
        assert by_name["coinapi_day_open_full"]["levels"] is None
        assert by_name["coinapi_stream_L5"] == {
            "name": "coinapi_stream_L5", "kind": "stream", "levels": 5}
        assert by_name["coinapi_on_demand_L20"]["kind"] == "on_demand"
        with pytest.raises(ValueError, match="unrecognized arm"):
            runner.arm_specs_from_names(["bogus_arm"])

    def test_full_day_gb_from_manifest(self, tmp_path):
        import importlib
        import json
        runner = importlib.import_module("scripts.run_snapshot_seed_experiment")
        mf = tmp_path / "_manifest.jsonl"
        rows = [{"dt": "2025-06-01", "status": "sample", "src_bytes": 1},
                # a TRADES row for the same date must never be taken as the BOOK size
                {"dt": "2025-06-01", "status": "ok", "src_bytes": 68_000_000,
                 "key": "T-TRADES/D-20250601/E-COINBASE/x+SC-COINBASE_SPOT_BTC_USD+.csv.gz"},
                {"dt": "2025-06-01", "status": "ok", "src_bytes": 799_598_234,
                 "key": "T-LIMITBOOK_FULL/D-20250601/E-COINBASE/x+SC-COINBASE_SPOT_BTC_USD+.csv.gz"},
                # legacy row without key/product fields still accepted
                {"dt": "2026-04-01", "status": "ok", "src_bytes": 2_371_844_307}]
        mf.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        gb, basis = runner.full_day_gb_from_manifest(mf, "2025-06-01")
        assert gb == pytest.approx(0.799598234)
        assert basis == "measured_src_bytes"
        gb_leg, _ = runner.full_day_gb_from_manifest(mf, "2026-04-01")
        assert gb_leg == pytest.approx(2.371844307)
        gb2, basis2 = runner.full_day_gb_from_manifest(mf, "1999-01-01")
        assert gb2 is None and basis2 == "missing"

    def test_reconcile_manifest_buckets_and_prices(self):
        import importlib
        rec = importlib.import_module("scripts.reconcile_snapshot_seed_54")
        manifest = {"cost_summary": {"gross_usd": 10.0, "book_usd": 9.0},
                    "days": [
            {"day": "2025-11-02", "classification": "inconclusive",
             "book_fill": {"needed": True, "kind": "full_day", "gb": 2.0, "usd": 2.0,
                           "full_day_reason": "crossed_seed_source", "why": "x"}},
            {"day": "2025-01-03", "classification": "missing_needs_coinapi",
             "book_fill": {"needed": True, "kind": "full_day", "gb": 1.0, "usd": 1.0,
                           "why": "missing_needs_coinapi"}},
            {"day": "2025-01-04", "classification": "",
             "book_fill": {"needed": True, "kind": "full_day", "gb": 1.2, "usd": 1.2,
                           "why": "calendar_book_gap"}},
            {"day": "2025-01-07", "classification": "lake_present_degraded",
             "book_fill": {"needed": True, "kind": "partial", "gb": 1.5, "usd": 1.5,
                           "why": "y"}},
            {"day": "2025-02-01", "classification": "lake_usable",
             "book_fill": {"needed": False}},
        ]}
        out = rec.reconcile_manifest(manifest, pilot_window=("2025-11-01", "2026-04-30"))
        b = out["buckets"]
        assert b["addressable_crossed_seed_source"]["days"] == 1
        assert b["addressable_crossed_seed_source"]["usd"] == 2.0
        assert b["addressable_crossed_seed_source"]["pilot_days"] == 1
        assert b["not_addressable_lake_absent"]["days"] == 2  # incl. calendar gap
        assert b["not_addressable_partial_fills"]["days"] == 1
        assert out["totals"]["book_fill_days"] == 4
        assert out["totals"]["gross_usd"] == 10.0

    def test_arm_summary_rows_carry_all_decision_bearing_fields(self):
        # The tracked evidence CSVs must let a reviewer validate EVERY number the
        # decision docs quote: reseed counts, seed acceptance, request counts and
        # termination, and the projected cost bands — not just the parity metrics.
        import importlib
        runner = importlib.import_module("scripts.run_snapshot_seed_experiment")
        rep = TestRunExperimentDay()._run()
        rows = {r["arm"]: r for r in runner.arm_summary_rows(rep)}
        st = rows["coinapi_stream_L2"]
        assert st["reseed_count"] is not None
        assert st["seed_accepted"] is True
        assert st["cost_usd_low"] is not None and st["cost_usd_high"] is not None
        assert st["full_day_usd"] is not None
        od = rows["coinapi_on_demand_L2"]
        assert od["n_requests"] >= 1 and od["terminated"]
        assert od["cost_usd_low"] is not None
        assert rows["cold_control"]["cost_usd_low"] is None  # unpriced control

    def test_missing_coinapi_parquet_exits_3(self, tmp_path, capsys):
        import importlib
        runner = importlib.import_module("scripts.run_snapshot_seed_experiment")
        rc = runner.main(["--day", "2025-06-01", "--coinapi-root", str(tmp_path),
                          "--lake-cache-root", str(tmp_path),
                          "--out-dir", str(tmp_path / "out"), "--engine", "python"])
        assert rc == 3
        assert "not found" in capsys.readouterr().err
