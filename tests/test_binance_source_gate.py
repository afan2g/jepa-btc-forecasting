"""Tests for the issue #64 Binance source-quality gate experiment.

Covers the preregistration pin, the CryptoHFTData adapter's fail-closed causal replay
(update-ID semantics, gaps, overlaps, resets, snapshot validity, lookahead rejection,
same-timestamp determinism), decimal/tick precision, frozen/silence/comparison metrics,
Stage-2 determinism comparison, the April forbidden-metric guard, network isolation, and
the fixture-file CLI paths (pyarrow-gated). NO network access anywhere: the adapter is
pure pandas/stdlib and the only network code (`cmd_fetch`) is exercised solely through its
refusal paths, which return before any socket is opened.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import binance_source_gate as bsg                             # noqa: E402

# ----------------------------------------------------------------------------- helpers
HOUR0 = int(pd.Timestamp("2026-04-01T12:00:00", tz="UTC").value)   # probe hour open (ns)
MS = 10**6
SEC = 10**9


def _cli():
    spec = importlib.util.spec_from_file_location(
        "run_binance_source_gate", str(ROOT / "scripts" / "run_binance_source_gate.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                         # dataclasses resolve __module__
    spec.loader.exec_module(mod)
    return mod


def snapshot_rows(last_update_id, t_ns, bids, asks, *, symbol="BTCUSDT"):
    """CommonOrderbookEvent snapshot rows (one per level), times in ms epoch."""
    rows = []
    for side, levels in (("bid", bids), ("ask", asks)):
        for price, qty in levels:
            rows.append({
                "received_time": t_ns // MS, "event_time": t_ns // MS,
                "transaction_time": None, "symbol": symbol, "event_type": "snapshot",
                "first_update_id": None, "final_update_id": None,
                "prev_final_update_id": None, "last_update_id": last_update_id,
                "side": side, "price": f"{price:.4f}", "quantity": f"{qty:.8f}"})
    return rows


def update_rows(first_u, final_u, prev_u, t_ns, levels, *, symbol="BTCUSDT"):
    """CommonOrderbookEvent update rows; `levels` = [(side, price, qty), ...]."""
    return [{
        "received_time": t_ns // MS, "event_time": t_ns // MS,
        "transaction_time": None, "symbol": symbol, "event_type": "update",
        "first_update_id": first_u, "final_update_id": final_u,
        "prev_final_update_id": prev_u, "last_update_id": None,
        "side": side, "price": f"{price:.4f}", "quantity": f"{qty:.8f}"}
        for side, price, qty in levels]


def chd_frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in ("first_update_id", "final_update_id", "prev_final_update_id",
                "last_update_id", "transaction_time"):
        df[col] = pd.array(df[col], dtype="Int64")
    return df


def five_levels(base_bid=100.0, base_ask=100.1, tick=0.1, qty=1.0):
    bids = [(round(base_bid - i * tick, 2), qty) for i in range(5)]
    asks = [(round(base_ask + i * tick, 2), qty) for i in range(5)]
    return bids, asks


def valid_hour(*, hour_ns=HOUR0, seed_id=1000):
    """A minimal valid futures hour: snapshot at +1s, contiguous updates after. The second
    update crosses the book (bid 100.2 > ask 100.1) for exactly one sample; the third
    removes the crossing bid again."""
    bids, asks = five_levels()
    rows = snapshot_rows(seed_id, hour_ns + 1 * SEC, bids, asks)
    rows += update_rows(seed_id - 2, seed_id, seed_id - 5, hour_ns + 500 * MS,
                        [("bid", 99.0, 1.0)])                     # pre-snapshot, skipped
    rows += update_rows(seed_id + 1, seed_id + 3, seed_id, hour_ns + 2 * SEC,
                        [("bid", 100.2, 2.0)])                    # anchors via pu == L
    rows += update_rows(seed_id + 4, seed_id + 6, seed_id + 3, hour_ns + 3 * SEC,
                        [("bid", 100.2, 0.0), ("ask", 100.3, 1.5)])
    return chd_frame(rows)


IDENTITY = {"exchange": "binance_futures", "symbol": "BTCUSDT", "date": "2026-04-01",
            "hour": 12}


def _identity(df):
    return bsg.validate_chd_frame(df, exchange=IDENTITY["exchange"],
                                  symbol=IDENTITY["symbol"], date_iso=IDENTITY["date"],
                                  hour=IDENTITY["hour"])


def grid(n=10, start=HOUR0, step=SEC):
    return [start + i * step for i in range(n)]


def replay(df, *, market="futures", scale=10, n=10, **kw):
    return bsg.replay_chd_window([(_identity(df), df)], market=market, price_scale=scale,
                                 grid=grid(n), **kw)


# ----------------------------------------------------------------------------- preregistration
class TestPreregistration:
    def test_artifact_loads_and_pins_module_constants(self):
        art = bsg.load_preregistration()
        assert art["issue"] == 64
        assert art["thresholds"] == bsg.PREREGISTERED["thresholds"]
        assert art["decision_logic"] == bsg.PREREGISTERED["decision_logic"]
        fx = bsg.PREREGISTERED["fixture_identity"]
        lake = art["fixture"]["lake"]
        assert lake["day"] == fx["lake_day"]
        assert lake["n_units"] == fx["lake_n_units"] == len(lake["units"])
        assert lake["rows_total"] == fx["lake_rows_total"]
        assert lake["out_bytes_total"] == fx["lake_out_bytes_total"]
        assert art["fixture"]["cryptohftdata"]["probe"]["object"] == fx["chd_probe_object"]

    def test_fixture_totals_reconcile_with_units(self):
        art = bsg.load_preregistration()
        units = art["fixture"]["lake"]["units"].values()
        assert sum(u["rows"] for u in units) == art["fixture"]["lake"]["rows_total"]
        assert sum(u["out_bytes"] for u in units) == art["fixture"]["lake"]["out_bytes_total"]
        assert all(len(u["sha256"]) == 64 for u in units)

    def test_amendments_append_only_structure(self):
        art = bsg.load_preregistration()
        assert isinstance(art["amendments"], list)
        for a in art["amendments"]:
            assert set(a) >= {"utc", "note"}

    def test_frozen_bars_match_production_thresholds(self):
        """The preregistered lake bars are the FROZEN production ones, not new numbers."""
        spec = importlib.util.spec_from_file_location(
            "rbr_for_prereg", str(ROOT / "scripts" / "run_binance_recon.py"))
        rbr = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = rbr                     # dataclasses resolve __module__
        spec.loader.exec_module(rbr)
        t = rbr.Thresholds()
        bars = bsg.PREREGISTERED["thresholds"]["lake_day_quality"]
        assert bars["crossed_usable_max"] == t.crossed_usable_max
        assert bars["missing_usable_max"] == t.missing_usable_max
        assert bars["thin_usable_max"] == t.thin_usable_max
        assert bars["seed_crossed_frac_max"] == t.seed_crossed_frac_max


# ----------------------------------------------------------------------------- units & decimals
class TestTimestampNormalization:
    def test_all_epoch_units_normalize_to_ns(self):
        t_ns = HOUR0 + 5 * SEC
        for div in (1, 10**3, 10**6, 10**9):
            arr = np.array([t_ns // div, t_ns // div + 1], dtype="int64")
            out = bsg.normalize_epoch_ns(arr, fieldname="event_time")
            assert out[0] == (t_ns // div) * div

    def test_out_of_range_epoch_refuses(self):
        with pytest.raises(bsg.ChdValidationError, match="timescale_undetectable"):
            bsg.normalize_epoch_ns(np.array([42], dtype="int64"), fieldname="event_time")
        with pytest.raises(bsg.ChdValidationError, match="timescale_undetectable"):
            bsg.normalize_epoch_ns(np.array([0, -5], dtype="int64"), fieldname="event_time")


class TestDecimalTicks:
    def test_decimal_places_normalized(self):
        assert bsg.decimal_places("50000.10") == 1
        assert bsg.decimal_places("50000.05") == 2
        assert bsg.decimal_places("50000") == 0
        assert bsg.decimal_places("0.000") == 0

    def test_to_ticks_exact_and_off_tick(self):
        assert bsg.to_ticks("50000.1", 10) == 500001
        assert bsg.to_ticks("50000.10", 10) == 500001
        with pytest.raises(bsg.ChdValidationError, match="off_tick"):
            bsg.to_ticks("50000.05", 10)

    def test_malformed_decimal_refuses(self):
        with pytest.raises(bsg.ChdValidationError, match="malformed_decimal"):
            bsg.decimal_places("not-a-price")
        with pytest.raises(bsg.ChdValidationError, match="malformed_decimal"):
            bsg.decimal_places("NaN")

    def test_measure_float_price_scale(self):
        clean = np.array([100.0, 100.1, 99.9, 250.5])
        m = bsg.measure_float_price_scale(clean, expected_decimals=1)
        assert m["ok"] and m["measured_decimals"] == 1 and m["conformance_scale"] == 10
        assert m["off_tick_at_expected"] == 0
        finer = np.array([100.0, 100.05])
        m2 = bsg.measure_float_price_scale(finer, expected_decimals=1)
        assert m2["ok"] and m2["measured_decimals"] == 2 and m2["conformance_scale"] == 100
        assert m2["off_tick_at_expected"] == 1
        bad = np.array([1.0 / 3.0])
        m3 = bsg.measure_float_price_scale(bad, expected_decimals=1)
        assert not m3["ok"] and m3["reason"] == "no_integral_scale"


# ----------------------------------------------------------------------------- validation
class TestChdValidate:
    def test_valid_frame_passes_with_identity(self):
        ident = _identity(valid_hour())
        assert ident["rows"] > 0
        assert ident["partition_axis"] in ("received_time", "event_time")
        assert ident["event_type_rows"]["snapshot"] == 10

    def test_missing_column_refuses(self):
        df = valid_hour().drop(columns=["prev_final_update_id"])
        with pytest.raises(bsg.ChdValidationError, match="schema_missing_columns"):
            _identity(df)

    def test_unknown_event_type_and_side_refuse(self):
        df = valid_hour()
        bad = df.copy()
        bad.loc[0, "event_type"] = "depthUpdate"
        with pytest.raises(bsg.ChdValidationError, match="unknown_event_type"):
            _identity(bad)
        bad = df.copy()
        bad.loc[0, "side"] = "buy"
        with pytest.raises(bsg.ChdValidationError, match="unknown_side"):
            _identity(bad)

    def test_wrong_symbol_refuses(self):
        df = valid_hour()
        df.loc[0, "symbol"] = "ETHUSDT"
        with pytest.raises(bsg.ChdValidationError, match="wrong_symbol"):
            _identity(df)

    def test_wrong_hour_refuses(self):
        df = valid_hour(hour_ns=HOUR0 + 3600 * SEC)      # rows actually in hour 13
        with pytest.raises(bsg.ChdValidationError, match="wrong_partition_window"):
            _identity(df)

    def test_empty_partition_refuses(self):
        df = valid_hour().iloc[0:0]
        with pytest.raises(bsg.ChdValidationError, match="empty_partition"):
            _identity(df)


# ----------------------------------------------------------------------------- causal replay
class TestChdReplayHappyPath:
    def test_snapshot_plus_contiguous_updates(self):
        frame, meta = replay(valid_hour())
        assert len(frame) == 10
        # sample at 12:00:00 precedes the snapshot (12:00:01) -> missing book
        assert np.isnan(frame.loc[0, "bid_0_price"])
        # sample at 12:00:01: snapshot applied at its own ts (apply-before-read)
        assert frame.loc[1, "bid_0_price"] == pytest.approx(100.0)
        assert frame.loc[1, "ask_0_price"] == pytest.approx(100.1)
        # 12:00:02: bid 100.2 added -> crossed vs ask 100.1 (recorded, not repaired)
        assert frame.loc[2, "bid_0_price"] == pytest.approx(100.2)
        # 12:00:03: crossing bid removed (qty 0) -> uncrossed again
        assert frame.loc[3, "bid_0_price"] == pytest.approx(100.0)
        assert frame.loc[3, "ask_0_price"] == pytest.approx(100.1)
        assert meta["counters"]["updates_applied"] == 2
        assert meta["counters"]["updates_skipped_pre_snapshot"] == 1
        assert meta["counters"]["snapshots_applied"] == 1
        assert meta["missing_book_samples"] == 1
        assert meta["crossed_samples"] == 1
        assert meta["frame_replay_hash"] is not None

    def test_internal_topk_contract_columns(self):
        frame, _ = replay(valid_hour(), k=2)
        assert list(frame.columns) == [
            "mid", "microprice", "bid_0_price", "bid_0_size", "ask_0_price", "ask_0_size",
            "bid_1_price", "bid_1_size", "ask_1_price", "ask_1_size", "sample_ts"]
        assert frame["sample_ts"].dtype == np.int64
        assert all(frame[c].dtype == np.float64 for c in frame.columns
                   if c != "sample_ts")

    def test_shuffled_rows_replay_identically(self):
        df = valid_hour()
        shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
        f1, m1 = replay(df)
        f2, m2 = replay(shuffled)
        assert m1["frame_replay_hash"] == m2["frame_replay_hash"]

    def test_same_timestamp_events_apply_in_update_id_order(self):
        bids, asks = five_levels()
        t = HOUR0 + 1 * SEC
        rows = snapshot_rows(1000, t, bids, asks)
        # two updates share event_time; the higher final_update_id must win the level
        rows += update_rows(1001, 1002, 1000, t + SEC, [("bid", 100.0, 5.0)])
        rows += update_rows(1003, 1004, 1002, t + SEC, [("bid", 100.0, 9.0)])
        frame, _ = replay(chd_frame(rows), n=4)
        assert frame.loc[3, "bid_0_size"] == pytest.approx(9.0)


class TestChdReplayFailClosed:
    def test_missing_initial_snapshot(self):
        rows = update_rows(1001, 1002, 1000, HOUR0 + SEC, [("bid", 100.0, 1.0)])
        with pytest.raises(bsg.ChdSnapshotError, match="missing_initial_snapshot"):
            replay(chd_frame(rows))

    def test_sequence_gap_futures(self):
        df = valid_hour()
        rows = df.to_dict("records")
        rows += update_rows(1010, 1012, 1008, HOUR0 + 4 * SEC, [("bid", 99.5, 1.0)])
        with pytest.raises(bsg.ChdContinuityError, match="sequence_gap"):
            replay(chd_frame(rows))

    def test_seed_anchor_gap_futures(self):
        bids, asks = five_levels()
        rows = snapshot_rows(1000, HOUR0 + SEC, bids, asks)
        # first post-snapshot update neither straddles L nor chains pu == L
        rows += update_rows(1005, 1007, 1004, HOUR0 + 2 * SEC, [("bid", 99.5, 1.0)])
        with pytest.raises(bsg.ChdContinuityError, match="seed_anchor_gap"):
            replay(chd_frame(rows))

    def test_futures_straddle_anchor_accepted(self):
        bids, asks = five_levels()
        rows = snapshot_rows(1000, HOUR0 + SEC, bids, asks)
        rows += update_rows(998, 1002, 995, HOUR0 + 2 * SEC, [("bid", 99.5, 1.0)])  # U<=L<=u
        _, meta = replay(chd_frame(rows))
        assert meta["counters"]["updates_applied"] == 1

    def test_spot_contiguity_chain(self):
        bids, asks = five_levels()
        rows = snapshot_rows(1000, HOUR0 + SEC, bids, asks)
        rows += update_rows(999, 1003, None, HOUR0 + 2 * SEC, [("bid", 99.95, 1.0)])
        rows += update_rows(1004, 1006, None, HOUR0 + 3 * SEC, [("bid", 99.94, 1.0)])
        _, meta = replay(chd_frame(rows), market="spot", scale=100)
        assert meta["counters"]["updates_applied"] == 2

    def test_spot_gap_refuses(self):
        bids, asks = five_levels()
        rows = snapshot_rows(1000, HOUR0 + SEC, bids, asks)
        rows += update_rows(999, 1003, None, HOUR0 + 2 * SEC, [("bid", 99.95, 1.0)])
        rows += update_rows(1006, 1008, None, HOUR0 + 3 * SEC, [("bid", 99.94, 1.0)])
        with pytest.raises(bsg.ChdContinuityError, match="sequence_gap"):
            replay(chd_frame(rows), market="spot", scale=100)

    def test_incompatible_overlap_once_anchored(self):
        """Two distinct events claiming the same final_update_id (partial overlap): the
        second does not advance the book version -> refused."""
        bids, asks = five_levels()
        rows = snapshot_rows(1000, HOUR0 + SEC, bids, asks)
        rows += update_rows(1001, 1003, 1000, HOUR0 + 2 * SEC, [("bid", 100.2, 2.0)])
        rows += update_rows(1002, 1003, 999, HOUR0 + 2 * SEC, [("bid", 99.5, 7.0)])
        with pytest.raises(bsg.ChdContinuityError, match="incompatible_overlap"):
            replay(chd_frame(rows))

    def test_same_ids_different_content_in_one_file_fails_closed(self):
        """A same-id re-capture with different payload/time cannot be grouped -> refused
        at grouping (fail closed) rather than silently merged."""
        df = valid_hour()
        rows = df.to_dict("records")
        rows += update_rows(1001, 1003, 1000, HOUR0 + 4 * SEC, [("bid", 99.5, 7.0)])
        with pytest.raises(bsg.ChdValidationError,
                           match="event_time_not_uniform|duplicate_level_in_event"):
            replay(chd_frame(rows))

    def test_exact_duplicate_event_deduped_across_files(self):
        bids, asks = five_levels()
        h12 = snapshot_rows(1000, HOUR0 + SEC, bids, asks)
        h12 += update_rows(1001, 1003, 1000, HOUR0 + 2 * SEC, [("bid", 100.2, 2.0)])
        dup = update_rows(1001, 1003, 1000, HOUR0 + 3600 * SEC + MS // MS,
                          [("bid", 100.2, 2.0)])
        h13 = dup + update_rows(1004, 1006, 1003, HOUR0 + 3601 * SEC, [("bid", 99.5, 1.0)])
        df12, df13 = chd_frame(h12), chd_frame(h13)
        i12 = _identity(df12)
        i13 = bsg.validate_chd_frame(df13, exchange="binance_futures", symbol="BTCUSDT",
                                     date_iso="2026-04-01", hour=13)
        frame, meta = bsg.replay_chd_window([(i12, df12), (i13, df13)], market="futures",
                                            price_scale=10, grid=grid(4))
        assert meta["counters"]["duplicate_events_dropped"] == 1
        assert meta["counters"]["updates_applied"] == 2

    def test_conflicting_duplicate_event_refuses(self):
        bids, asks = five_levels()
        h12 = snapshot_rows(1000, HOUR0 + SEC, bids, asks)
        h12 += update_rows(1001, 1003, 1000, HOUR0 + 2 * SEC, [("bid", 100.2, 2.0)])
        h13 = update_rows(1001, 1003, 1000, HOUR0 + 3600 * SEC + SEC,
                          [("bid", 100.2, 999.0)])       # same ids, different content
        df12, df13 = chd_frame(h12), chd_frame(h13)
        i13 = bsg.validate_chd_frame(df13, exchange="binance_futures", symbol="BTCUSDT",
                                     date_iso="2026-04-01", hour=13)
        with pytest.raises(bsg.ChdContinuityError, match="conflicting_duplicate_event"):
            bsg.replay_chd_window([(_identity(df12), df12), (i13, df13)], market="futures",
                                  price_scale=10, grid=grid(4))

    def test_reset_snapshot_replaces_state_and_rearms_anchor(self):
        bids, asks = five_levels()
        rows = snapshot_rows(1000, HOUR0 + SEC, bids, asks)
        rows += update_rows(1001, 1003, 1000, HOUR0 + 2 * SEC, [("bid", 100.2, 2.0)])
        nb, na = five_levels(base_bid=200.0, base_ask=200.1)
        rows += snapshot_rows(2000, HOUR0 + 4 * SEC, nb, na)
        rows += update_rows(1998, 2002, 1990, HOUR0 + 5 * SEC, [("bid", 199.9, 3.0)])
        frame, meta = replay(chd_frame(rows))
        assert meta["counters"]["resets"] == 1
        assert frame.loc[4, "bid_0_price"] == pytest.approx(200.0)   # old 100.2 dropped
        assert meta["counters"]["updates_applied"] == 2

    def test_stale_backwards_snapshot_refuses_across_files(self):
        """A next-hour snapshot carrying an OLDER book version than the applied state must
        never reseed (backwards/stale book version)."""
        bids, asks = five_levels()
        h12 = snapshot_rows(1000, HOUR0 + SEC, bids, asks)
        h12 += update_rows(1001, 1003, 1000, HOUR0 + 2 * SEC, [("bid", 100.2, 2.0)])
        h13 = snapshot_rows(900, HOUR0 + 3601 * SEC, bids, asks)     # older book version
        df12, df13 = chd_frame(h12), chd_frame(h13)
        i13 = bsg.validate_chd_frame(df13, exchange="binance_futures", symbol="BTCUSDT",
                                     date_iso="2026-04-01", hour=13)
        with pytest.raises(bsg.ChdSnapshotError, match="stale_snapshot"):
            bsg.replay_chd_window([(_identity(df12), df12), (i13, df13)],
                                  market="futures", price_scale=10, grid=grid(4))

    def test_backwards_snapshot_within_file_is_an_ordering_anomaly(self):
        """Within one ID-ordered file, a late-timestamped low-ID snapshot surfaces as an
        event_time ordering anomaly -> refused (fail closed, different reason code)."""
        bids, asks = five_levels()
        rows = snapshot_rows(1000, HOUR0 + SEC, bids, asks)
        rows += update_rows(1001, 1003, 1000, HOUR0 + 2 * SEC, [("bid", 100.2, 2.0)])
        rows += snapshot_rows(900, HOUR0 + 3 * SEC, bids, asks)      # sorts FIRST by id
        with pytest.raises(bsg.ChdContinuityError, match="ordering_anomaly"):
            replay(chd_frame(rows))

    def test_snapshot_never_applies_before_its_own_time(self):
        """Lookahead rejection: samples before the seed's event_time stay missing."""
        bids, asks = five_levels()
        rows = snapshot_rows(1000, HOUR0 + 5 * SEC, bids, asks)
        frame, _ = replay(chd_frame(rows), n=8)
        assert frame.loc[:4, "bid_0_price"].isna().all()
        assert frame.loc[5, "bid_0_price"] == pytest.approx(100.0)

    def test_truncated_one_sided_and_crossed_snapshots_refuse(self):
        thin_b = [(100.0 - i * 0.1, 1.0) for i in range(3)]
        thin_a = [(100.1 + i * 0.1, 1.0) for i in range(3)]
        with pytest.raises(bsg.ChdSnapshotError, match="snapshot_thin_depth"):
            replay(chd_frame(snapshot_rows(1000, HOUR0 + SEC, thin_b, thin_a)))
        bids, asks = five_levels()
        with pytest.raises(bsg.ChdSnapshotError, match="snapshot_one_sided"):
            replay(chd_frame(snapshot_rows(1000, HOUR0 + SEC, bids, [])))
        cb = [(100.2 - i * 0.1, 1.0) for i in range(5)]              # best bid 100.2
        ca = [(100.1 + i * 0.1, 1.0) for i in range(5)]              # best ask 100.1
        with pytest.raises(bsg.ChdSnapshotError, match="snapshot_crossed"):
            replay(chd_frame(snapshot_rows(1000, HOUR0 + SEC, cb, ca)))

    def test_zero_size_delete_absent_is_counted_noop(self):
        df = valid_hour()
        rows = df.to_dict("records")
        rows += update_rows(1007, 1008, 1006, HOUR0 + 4 * SEC, [("bid", 55.5, 0.0)])
        _, meta = replay(chd_frame(rows))
        assert meta["counters"]["delete_absent_levels"] == 1

    def test_duplicate_level_in_one_event_refuses(self):
        bids, asks = five_levels()
        rows = snapshot_rows(1000, HOUR0 + SEC, bids, asks)
        rows += update_rows(1001, 1002, 1000, HOUR0 + 2 * SEC,
                            [("bid", 100.0, 1.0), ("bid", 100.0, 2.0)])
        with pytest.raises(bsg.ChdValidationError, match="duplicate_level_in_event"):
            replay(chd_frame(rows))

    def test_duplicate_rows_within_file_deduped(self):
        df = valid_hour()
        doubled = pd.concat([df, df.iloc[[len(df) - 1]]], ignore_index=True)
        f1, m1 = replay(valid_hour())
        f2, m2 = replay(doubled)
        assert m2["counters"]["duplicate_rows_dropped"] == 1
        assert m1["frame_replay_hash"] == m2["frame_replay_hash"]

    def test_event_time_regression_bound_refuses(self):
        bids, asks = five_levels()
        rows = snapshot_rows(1000, HOUR0 + 5 * SEC, bids, asks)
        # a later-id update whose event_time regresses far behind the watermark
        rows += update_rows(1001, 1002, 1000, HOUR0 + 1 * SEC, [("bid", 99.5, 1.0)])
        with pytest.raises(bsg.ChdContinuityError, match="ordering_anomaly"):
            replay(chd_frame(rows))

    def test_missing_and_duplicate_hour_partitions_refuse(self):
        i12 = _identity(valid_hour())
        i14 = dict(i12, hour=14)
        with pytest.raises(bsg.ChdValidationError, match="missing_hour_partition"):
            bsg.require_consecutive_hours([i12, i14])
        with pytest.raises(bsg.ChdValidationError, match="duplicate_hour_partition"):
            bsg.require_consecutive_hours([i12, dict(i12)])
        bsg.require_consecutive_hours([i12, dict(i12, hour=13)])     # consecutive: fine


# ----------------------------------------------------------------------------- metrics
class TestFrozenSilence:
    def _static_frame(self, n, *, bid=100.0, ask=100.1, k=1):
        rows = []
        for i in range(n):
            rows.append({"mid": (bid + ask) / 2, "microprice": (bid + ask) / 2,
                         "bid_0_price": bid, "bid_0_size": 1.0,
                         "ask_0_price": ask, "ask_0_size": 1.0,
                         "sample_ts": HOUR0 + i * SEC})
        return pd.DataFrame(rows)

    def test_frozen_run_detected_at_60s(self):
        f = self._static_frame(60)
        m = bsg.frozen_metrics(f)
        assert m["n_frozen_runs"] == 1 and m["frozen_fraction"] == 1.0
        assert m["stale_but_uncrossed_fraction"] == 1.0

    def test_run_below_60s_not_frozen(self):
        f = self._static_frame(59)
        m = bsg.frozen_metrics(f)
        assert m["n_frozen_runs"] == 0 and m["frozen_fraction"] == 0.0

    def test_changing_book_not_frozen(self):
        f = self._static_frame(120)
        f.loc[::2, "bid_0_size"] = 2.0
        m = bsg.frozen_metrics(f)
        assert m["n_frozen_runs"] == 0

    def test_silence_metrics(self):
        t = np.array([0, 5, 20, 400], dtype="int64") * SEC
        m = bsg.silence_metrics(t)
        assert m["max_gap_s"] == 380.0
        assert m["gaps_gt_10s"] == 2 and m["gaps_gt_300s"] == 1
        assert m["silent_seconds_gt_10s"] == pytest.approx(395.0)


class TestComparison:
    def _frame(self, n=100, *, bid=100.0, ask=100.1):
        rows = []
        for i in range(n):
            r = {"mid": (bid + ask) / 2, "microprice": (bid + ask) / 2}
            for j in range(10):
                r[f"bid_{j}_price"] = round(bid - 0.1 * j, 4)
                r[f"bid_{j}_size"] = 1.0
                r[f"ask_{j}_price"] = round(ask + 0.1 * j, 4)
                r[f"ask_{j}_size"] = 1.0
            r["sample_ts"] = HOUR0 + i * SEC
            rows.append(r)
        return pd.DataFrame(rows)

    def test_identical_frames_pass_all_bars(self):
        a, b = self._frame(), self._frame()
        m = bsg.compare_topk_frames(a, b, price_scale=10)
        assert m["joint_valid_fraction"] == 1.0
        assert m["touch_agreement_exact_tick"] == 1.0
        assert m["mid_abs_diff_ticks"]["p99"] == 0.0
        assert m["topk_shared_levels_mean"]["bid"] == 10.0
        ev = bsg.evaluate_comparison(m)
        assert ev["pass"] and all(c["ok"] for c in ev["checks"])

    def test_one_tick_disagreement_stays_within_bars(self):
        a, b = self._frame(), self._frame()
        b.loc[:4, "ask_0_price"] += 0.1                  # 5% of samples off by one tick
        m = bsg.compare_topk_frames(a, b, price_scale=10)
        assert m["touch_agreement_exact_tick"] == pytest.approx(0.95)
        assert m["touch_agreement_within_1_tick"] == 1.0
        assert bsg.evaluate_comparison(m)["pass"]

    def test_grid_mismatch_refuses(self):
        a, b = self._frame(), self._frame()
        b["sample_ts"] += SEC
        with pytest.raises(bsg.SourceGateError, match="grid_mismatch"):
            bsg.compare_topk_frames(a, b, price_scale=10)

    def test_missing_metrics_fail_closed(self):
        a = self._frame(10)
        b = self._frame(10)
        b.loc[:, [f"bid_{j}_price" for j in range(10)]] = np.nan   # never joint-valid
        m = bsg.compare_topk_frames(a, b, price_scale=10)
        assert m["n_joint_valid"] == 0
        ev = bsg.evaluate_comparison(m)
        assert not ev["pass"]

    def test_off_tick_frame_refuses(self):
        a, b = self._frame(), self._frame()
        b.loc[0, "ask_0_price"] = 100.1234
        with pytest.raises(bsg.SourceGateError, match="off_tick"):
            bsg.compare_topk_frames(a, b, price_scale=10)


# ----------------------------------------------------------------------------- determinism cmp
class TestStage2Determinism:
    def _write_manifest(self, path, rows_a=100, secs=1.0):
        recs = [
            {"output": "topk_l2", "exchange": "BINANCE_FUTURES", "symbol": "BTC-USDT-PERP",
             "dt": "2026-04-01", "status": "ok", "classification": "certified",
             "rows": rows_a, "sha256": "a" * 64, "secs": secs, "ts": f"t{secs}"},
            {"output": "trades", "exchange": "BINANCE_FUTURES", "symbol": "BTC-USDT-PERP",
             "dt": "2026-04-01", "status": "ok", "rows": 5, "sha256": "b" * 64,
             "secs": secs * 2, "ts": f"u{secs}"},
        ]
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

    def test_equal_modulo_volatile_keys(self, tmp_path):
        p1, p2 = tmp_path / "m1.jsonl", tmp_path / "m2.jsonl"
        self._write_manifest(p1, secs=1.0)
        self._write_manifest(p2, secs=9.9)               # secs/ts differ -> still equal
        out = bsg.compare_stage2_manifests(str(p1), str(p2))
        assert out["equal"] and out["n_units"] == 2

    def test_semantic_difference_detected(self, tmp_path):
        p1, p2 = tmp_path / "m1.jsonl", tmp_path / "m2.jsonl"
        self._write_manifest(p1, rows_a=100)
        self._write_manifest(p2, rows_a=101)
        out = bsg.compare_stage2_manifests(str(p1), str(p2))
        assert not out["equal"]
        assert any("rows" in d["diff"] for d in out["diffs"])


# ----------------------------------------------------------------------------- April guard
class TestAprilGuard:
    def test_forbidden_keys_refuse(self):
        for key in ("label_agreement", "pnl_total", "feature_matrix", "cost_bps",
                    "trade_notional_sum", "interarrival_p50", "mid_usd_mean",
                    "return_1s", "realized_volatility", "price_path_excerpt"):
            with pytest.raises(bsg.SourceGateError, match="forbidden_metric"):
                bsg.assert_report_publishable({"metrics": {key: 1.0}})

    def test_unbounded_series_refuses(self):
        with pytest.raises(bsg.SourceGateError, match="unbounded_series"):
            bsg.assert_report_publishable({"series": list(range(500))})

    def test_clean_report_passes_and_hashes_deterministically(self):
        rep = {"step": "x", "crossed_rate": 0.001, "counters": {"events": 10},
               "nested": [{"ok": True}]}
        out1 = bsg.finalize_report(dict(rep))
        out2 = bsg.finalize_report(dict(rep))
        assert out1["report_hash"] == out2["report_hash"]
        assert len(out1["report_hash"]) == 64

    def test_replay_meta_passes_guard(self):
        _, meta = replay(valid_hour())
        bsg.assert_report_publishable(json.loads(json.dumps(bsg._json_safe(meta))))


# ----------------------------------------------------------------------------- verdict CLI
class TestVerdictCli:
    UNITS = {
        ("BINANCE_FUTURES", "BTC-USDT-PERP"): ["topk_l2", "trades", "funding",
                                               "open_interest", "liquidations"],
        ("BINANCE", "BTC-USDT"): ["topk_l2", "trades"],
    }

    def _stage2_manifest(self, path, *, perp_topk_cls="certified"):
        with open(path, "w") as f:
            for (exchange, symbol), outputs in self.UNITS.items():
                for output in outputs:
                    rec = {"output": output, "exchange": exchange, "symbol": symbol,
                           "dt": "2026-04-01", "status": "ok", "rows": 1,
                           "sha256": "c" * 64}
                    if output == "topk_l2":
                        cls = perp_topk_cls if exchange == "BINANCE_FUTURES" \
                            else "certified"
                        rec["classification"] = cls
                        if cls != "certified":
                            rec["status"] = cls if cls == "inconclusive" else "ok"
                            if cls == "inconclusive":
                                rec["status"] = "inconclusive"
                                rec["rows"] = 0
                    f.write(json.dumps(rec) + "\n")

    def _step_reports(self, tmp_path, *, determinism_pass=True, tick_pass=True):
        paths = {}
        for name, payload in {
            "verify": {"step": "verify-inputs", "pass": True},
            "tick": {"step": "tick-scale", "pass": tick_pass},
            "silence": {"step": "silence", "pass": True},
            "det": {"step": "stage2-compare", "pass": determinism_pass},
        }.items():
            p = tmp_path / f"{name}.json"
            p.write_text(json.dumps(payload))
            paths[name] = str(p)
        for inst in ("binance-perp", "binance-spot"):
            p = tmp_path / f"replay_{inst}.json"
            p.write_text(json.dumps({
                "step": "replay-conformance", "instrument": inst, "pass": True,
                "harness_determinism_ok": True, "conformance_ok": True,
                "conformance": {"ran": True},
                "frozen": {"frozen_cap_fired": False, "frozen_fraction": 0.0}}))
            paths[inst] = str(p)
        return paths

    def _run(self, tmp_path, manifest, paths):
        cli = _cli()
        rc = cli.main(["verdict", "--stage2-manifest", str(manifest),
                       "--verify-report", paths["verify"], "--tick-report", paths["tick"],
                       "--silence-report", paths["silence"],
                       "--determinism-report", paths["det"],
                       "--replay-report", paths["binance-perp"],
                       "--replay-report", paths["binance-spot"],
                       "--out", str(tmp_path)])
        assert rc == 0
        return json.loads((tmp_path / "lake_verdict.json").read_text())

    def test_all_green_is_certified(self, tmp_path):
        m = tmp_path / "m.jsonl"
        self._stage2_manifest(m)
        rep = self._run(tmp_path, m, self._step_reports(tmp_path))
        assert rep["lake_verdict"] == "certified"

    def test_degraded_topk_or_cap_is_degraded(self, tmp_path):
        m = tmp_path / "m.jsonl"
        self._stage2_manifest(m, perp_topk_cls="degraded")
        rep = self._run(tmp_path, m, self._step_reports(tmp_path))
        assert rep["lake_verdict"] == "degraded"
        m2 = tmp_path / "m2.jsonl"
        self._stage2_manifest(m2)
        rep = self._run(tmp_path, m2, self._step_reports(tmp_path, tick_pass=False))
        assert rep["lake_verdict"] == "degraded"

    def test_hard_invalidator_is_inconclusive(self, tmp_path):
        m = tmp_path / "m.jsonl"
        self._stage2_manifest(m)
        rep = self._run(tmp_path, m, self._step_reports(tmp_path, determinism_pass=False))
        assert rep["lake_verdict"] == "inconclusive"

    def test_inconclusive_topk_is_inconclusive(self, tmp_path):
        m = tmp_path / "m.jsonl"
        self._stage2_manifest(m, perp_topk_cls="inconclusive")
        rep = self._run(tmp_path, m, self._step_reports(tmp_path))
        assert rep["lake_verdict"] == "inconclusive"


# ----------------------------------------------------------------------------- network isolation
class TestNetworkIsolation:
    def _module_level_imports(self, path):
        tree = ast.parse(pathlib.Path(path).read_text())
        names = set()
        for node in tree.body:                           # module level only
            if isinstance(node, ast.Import):
                names |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                names.add(node.module or "")
        return names

    def test_experiment_module_has_no_network_imports(self):
        names = self._module_level_imports(ROOT / "experiments" / "binance_source_gate.py")
        for banned in ("urllib", "requests", "socket", "http", "boto3", "lakeapi"):
            assert not any(n == banned or n.startswith(banned + ".") for n in names), banned

    def test_cli_module_has_no_network_imports_at_top(self):
        names = self._module_level_imports(ROOT / "scripts" / "run_binance_source_gate.py")
        for banned in ("urllib", "requests", "socket", "http", "boto3", "lakeapi",
                       "pyarrow"):
            assert not any(n == banned or n.startswith(banned + ".") for n in names), banned

    def test_fetch_refuses_without_approval_before_any_network(self, tmp_path):
        cli = _cli()
        rc = cli.main(["fetch", "--object", "binance_futures/2026-04-01/12/x.parquet.zst",
                       "--dest", str(tmp_path / "x.zst"), "--out", str(tmp_path)])
        assert rc == cli.SETUP_ERROR_EXIT

    def test_fetch_refuses_existing_dest(self, tmp_path):
        cli = _cli()
        dest = tmp_path / "x.zst"
        dest.write_bytes(b"do-not-overwrite")
        rc = cli.main(["fetch", "--object", "o", "--dest", str(dest),
                       "--approved-by", "user", "--out", str(tmp_path)])
        assert rc == cli.SETUP_ERROR_EXIT
        assert dest.read_bytes() == b"do-not-overwrite"


# ----------------------------------------------------------------------------- fixture-file CLI
class TestCliWithFixtures:
    pa = pytest.importorskip("pyarrow")

    def _write_hour(self, tmp_path, df, name="BTCUSDT_orderbook.parquet", zstd_outer=False):
        import pyarrow as pa
        import pyarrow.parquet as pq
        plain = tmp_path / name
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), plain)
        if not zstd_outer:
            return str(plain)
        z = tmp_path / (name + ".zst")
        with open(plain, "rb") as src, pa.output_stream(str(z), compression="zstd") as out:
            out.write(src.read())
        plain.unlink()
        return str(z)

    def test_chd_validate_cli_pass_and_refusal(self, tmp_path):
        cli = _cli()
        path = self._write_hour(tmp_path, valid_hour(), zstd_outer=True)
        rc = cli.main(["chd-validate", "--file", path, "--exchange", "binance_futures",
                       "--date", "2026-04-01", "--hour", "12", "--out", str(tmp_path)])
        assert rc == 0
        rep = json.loads((tmp_path / "chd_validate_binance_futures_2026-04-01_12.json")
                         .read_text())
        assert rep["pass"] and rep["identity"]["provenance"]["compression"] == "zstd"
        rc = cli.main(["chd-validate", "--file", path, "--exchange", "binance_futures",
                       "--date", "2026-04-02", "--hour", "12", "--out", str(tmp_path)])
        assert rc == cli.FAIL_EXIT
        rep = json.loads((tmp_path / "chd_validate_binance_futures_2026-04-02_12.json")
                         .read_text())
        assert not rep["pass"] and rep["refusal"] == "wrong_partition_window"

    def test_chd_replay_cli_certified_and_frame_out(self, tmp_path):
        import pyarrow.parquet as pq
        cli = _cli()
        # dense hour: 10-level snapshot (not thin at k=10) at +1s, then a small contiguous
        # update every second
        bids = [(round(100.0 - i * 0.1, 2), 1.0) for i in range(10)]
        asks = [(round(100.1 + i * 0.1, 2), 1.0) for i in range(10)]
        rows = snapshot_rows(1000, HOUR0 + SEC, bids, asks)
        u = 1000
        for i in range(2, 3600):
            rows += update_rows(u + 1, u + 2, u, HOUR0 + i * SEC,
                                [("bid", 99.6, 1.0 + (i % 3))])
            u += 2
        path = self._write_hour(tmp_path, chd_frame(rows))
        frame_out = tmp_path / "chd_frame.parquet"
        rc = cli.main(["chd-replay", "--files", path, "--exchange", "binance_futures",
                       "--date", "2026-04-01", "--start-hour", "12", "--n-hours", "1",
                       "--scale", "10", "--out", str(tmp_path),
                       "--frame-out", str(frame_out)])
        assert rc == 0
        rep = json.loads((tmp_path / "chd_replay_binance_futures_2026-04-01_12_1h.json")
                         .read_text())
        assert rep["chd_verdict"] == "certified" and rep["pass"]
        assert rep["meta"]["missing_book_fraction"] <= 0.02
        with pq.ParquetFile(str(frame_out)) as pf:
            assert pf.metadata.num_rows == 3600

    def test_chd_replay_cli_fail_closed_writes_refusal_report(self, tmp_path):
        cli = _cli()
        rows = update_rows(1001, 1002, 1000, HOUR0 + SEC, [("bid", 100.0, 1.0)])
        path = self._write_hour(tmp_path, chd_frame(rows))
        rc = cli.main(["chd-replay", "--files", path, "--exchange", "binance_futures",
                       "--date", "2026-04-01", "--start-hour", "12", "--n-hours", "1",
                       "--scale", "10", "--out", str(tmp_path)])
        assert rc == cli.FAIL_EXIT
        rep = json.loads((tmp_path / "chd_replay_binance_futures_2026-04-01_12_1h.json")
                         .read_text())
        assert rep["chd_verdict"] == "inconclusive"
        assert rep["refusal"] == "missing_initial_snapshot"
        assert not (tmp_path / "chd_frame.parquet").exists()

    def test_compare_cli_on_identical_frames(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq
        cli = _cli()
        f = TestComparison()._frame(50)
        for name in ("lake.parquet", "chd.parquet"):
            pq.write_table(pa.Table.from_pandas(f, preserve_index=False),
                           str(tmp_path / name))
        rc = cli.main(["compare", "--lake-frame", str(tmp_path / "lake.parquet"),
                       "--chd-frame", str(tmp_path / "chd.parquet"), "--scale", "10",
                       "--out", str(tmp_path)])
        assert rc == 0
        rep = json.loads((tmp_path / "comparison.json").read_text())
        assert rep["pass"] and rep["evaluation"]["pass"]

    def test_frame_replay_hash_pins_to_snapshot_seed_implementation(self):
        from experiments.snapshot_seed import frame_replay_hash as h54
        frame, _ = replay(valid_hour())
        assert bsg.frame_replay_hash(frame) == h54(frame)
