"""T9 producer orchestration tests — development mode (issue #94).

Synthetic/tiny fixtures only (tests/produce_fixtures.py); no vendor I/O and no
real January access. Holdout/blind-materializer coverage lives in
tests/test_produce_holdout.py.
"""
from __future__ import annotations

import pytest

from bars.clock import ThresholdConfig
from bars.events import ClockTrade
from bars.modes import (
    BINANCE_SINGLE_VENUE,
    COINBASE_ONLY,
    CROSS_VENUE,
    SOURCE_MODES,
    allowed_venues,
    require_venue_allowed,
)
from bars.produce import (
    DROP_COUNT_CATEGORIES,
    RuntimeParams,
    iter_normalized_book_events,
    produce_development,
    read_normalized_trades,
)
from eval.g0bn_identity import development_data_identity
from eval.writer import classify_manifest
from eval.manifest import load_manifest
from tests.g0bn_dev_fixtures import (
    dev_config,
    included_days as dev_included_days,
    runtime_cv,
)
from tests.g0bn_protocol_fixtures import (
    make_clock,
    make_exclusions,
    make_features,
    make_producer,
)
from tests.produce_fixtures import (
    SEED_THRESHOLD,
    TARGET_BARS_PER_DAY,
    TIME_CAP_NS,
    SyntheticWorld,
    day_open_ns,
    write_day_objects,
)

GEN_AT = "2026-07-19T00:00:00Z"


def make_runtime(**over) -> RuntimeParams:
    d = dict(
        threshold=ThresholdConfig(
            target_bars_per_day=TARGET_BARS_PER_DAY, window_days=3, warmup_days=1,
            seed_threshold=SEED_THRESHOLD, min_covered_fraction=0.0),
        top_k=3, tick_size=0.01, min_returns=2, vol_floor_bps=0.25)
    d.update(over)
    return RuntimeParams(**d)


def produce_config(**over):
    clock = over.pop("clock", None)
    if clock is None:
        clock = make_clock(target_bars_per_day=TARGET_BARS_PER_DAY,
                           time_cap_ns=TIME_CAP_NS)
    return dev_config(clock=clock, **over)


# ------------------------------------------------------------------ source modes


def test_source_modes_include_binance_single_venue():
    assert SOURCE_MODES == (COINBASE_ONLY, CROSS_VENUE, BINANCE_SINGLE_VENUE)
    assert BINANCE_SINGLE_VENUE == "binance_single_venue"


def test_binance_single_venue_allows_only_binance():
    assert allowed_venues(BINANCE_SINGLE_VENUE) == ("binance",)
    require_venue_allowed(BINANCE_SINGLE_VENUE, "binance")
    with pytest.raises(ValueError, match="does not allow"):
        require_venue_allowed(BINANCE_SINGLE_VENUE, "coinbase")


def test_legacy_modes_unchanged():
    assert allowed_venues(COINBASE_ONLY) == ("coinbase",)
    assert allowed_venues(CROSS_VENUE) == ("coinbase", "binance")


# ------------------------------------------------------- normalized object readers


def test_read_normalized_trades_contract(tmp_path):
    world = SyntheticWorld()
    paths, _ = write_day_objects(world, "2025-11-01", tmp_path)
    trades = read_normalized_trades(paths["binance_futures_trades"])
    assert trades and all(isinstance(t, ClockTrade) for t in trades)
    assert all(t.received_time >= t.origin_time for t in trades)
    assert {t.side for t in trades} == {"buy", "sell"}


def test_iter_normalized_book_events_streams_in_order(tmp_path):
    world = SyntheticWorld()
    paths, _ = write_day_objects(world, "2025-11-01", tmp_path)
    events = list(iter_normalized_book_events(paths["binance_futures_l2_delta"]))
    assert events
    keys = [(e.origin_time, e.seq) for e in events]
    assert keys == sorted(keys)
    assert {e.side for e in events} == {"bid", "ask"}


def test_iter_normalized_book_events_fails_closed_on_disorder(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "bad.parquet"
    t0 = 1_800_000_000_000_000_000
    pq.write_table(pa.table({
        "origin_time": pa.array([t0 + 5, t0 + 1], pa.int64()),
        "received_time": pa.array([t0 + 6, t0 + 2], pa.int64()),
        "seq": pa.array([2, 1], pa.int64()),
        "side": pa.array(["bid", "ask"]),
        "price": pa.array([99.0, 101.0], pa.float64()),
        "size": pa.array([1.0, 1.0], pa.float64()),
    }), path)
    with pytest.raises(ValueError, match="order"):
        list(iter_normalized_book_events(path))


def test_l2_reader_rejects_non_absolute_timestamps(tmp_path):
    # Codex round 16: a time-of-day or wrong-unit L2 timestamp (e.g. origin=1)
    # would pass the seed's at-or-before-open check and seed books/labels; the
    # reader must enforce the same absolute-UTC floor the trades path uses
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "offset.parquet"
    pq.write_table(pa.table({
        "origin_time": pa.array([1], pa.int64()),
        "received_time": pa.array([2], pa.int64()),
        "seq": pa.array([1], pa.int64()),
        "side": pa.array(["bid"]),
        "price": pa.array([99.0], pa.float64()),
        "size": pa.array([5.0], pa.float64()),
    }), path)
    with pytest.raises(ValueError, match="absolute"):
        list(iter_normalized_book_events(path))


def test_read_normalized_trades_rejects_receipt_before_origin(tmp_path):
    # Codex round 7: a trade captured "before" it happened breaks the two-axis
    # contract and would understate the monotone decision watermark (lookahead);
    # the trade reader must enforce received_time >= origin_time like the L2 one
    import pyarrow as pa
    import pyarrow.parquet as pq

    t0 = 1_800_000_000_000_000_000
    path = tmp_path / "trades.parquet"
    pq.write_table(pa.table({
        "origin_time": pa.array([t0], pa.int64()),
        "received_time": pa.array([t0 - 1], pa.int64()),
        "seq": pa.array([1], pa.int64()),
        "side": pa.array(["buy"]),
        "price": pa.array([100.0], pa.float64()),
        "quantity": pa.array([0.5], pa.float64()),
    }), path)
    with pytest.raises(ValueError, match="received_time"):
        read_normalized_trades(path)


def test_read_normalized_trades_rejects_missing_column(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "trades.parquet"
    pq.write_table(pa.table({"origin_time": pa.array([1], pa.int64())}), path)
    with pytest.raises(ValueError):
        read_normalized_trades(path)


# ------------------------------------------------------------- development build


@pytest.fixture(scope="module")
def dev_build(tmp_path_factory):
    root = tmp_path_factory.mktemp("dev-build")
    world = SyntheticWorld()
    config = produce_config()
    days = ("2025-11-01", "2025-11-02", "2025-11-03")
    day_objects, day_shas = {}, {}
    for day in days:
        paths, shas = write_day_objects(world, day, root)
        day_objects[day] = paths
        day_shas[day] = shas
    result = produce_development(
        config, runtime=make_runtime(), day_objects=day_objects,
        matrix_path=root / "model_matrix.parquet",
        manifest_path=root / "feature_manifest.json",
        generated_at=GEN_AT)
    return {"config": config, "root": root, "result": result,
            "day_objects": day_objects, "day_shas": day_shas}


def test_development_build_publishes_valid_g0bn_artifacts(dev_build):
    import pandas as pd

    result = dev_build["result"]
    manifest = load_manifest(result.write.manifest_path)
    cls = classify_manifest(manifest)
    assert cls.is_g0bn and cls.partition == "development" and not cls.holdout_bound
    frame = pd.read_parquet(result.write.matrix_path)
    assert len(frame) == result.write.row_count > 0
    # every declared horizon survives on supported synthetic days
    assert set(frame["horizon"]) == {"2s", "10s", "60s"}
    # synchronous decide-and-act timing
    assert (frame["t_available"] == frame["t_event"]).all()
    # binary64 cost diagnostics as separate non-feature columns
    for col in ("cost_bps", "half_spread_bps", "latency_drift_bps"):
        assert str(frame[col].dtype) == "float64"
    assert (frame["latency_drift_bps"] >= 0).all()
    # the varying capture lag makes the observable and true reads genuinely
    # differ on some bars: a wiring bug collapsing the two cuts (or destroying
    # the received gate) would zero the drift everywhere
    assert (frame["latency_drift_bps"] > 0).any()
    assert frame.duplicated(["t_event", "horizon"]).sum() == 0


def test_development_drop_counts_cover_the_pinned_categories(dev_build):
    counts = dev_build["result"].drop_counts
    assert tuple(counts) == DROP_COUNT_CATEGORIES
    for cat, per_tag in counts.items():
        assert set(per_tag) == {"2s", "10s", "60s"}
        assert all(isinstance(n, int) and n >= 0 for n in per_tag.values())
    # warm-up day 1 is dropped by the trailing schedule contract
    assert all(n > 0 for n in counts["warmup"].values())
    # quiet-tail time-cap bars reject on the staleness gate
    assert all(n > 0 for n in counts["staleness"].values())
    # day 1's no_prior_read bar is masked by warmup (first failure wins); the
    # feature_rejection path is asserted in the holdout tests, where January
    # day 1 is not a warm-up day thanks to the frozen development-end state
    assert all(n == 0 for n in counts["feature_rejection"].values())


def test_development_rebuild_is_logically_identical(dev_build, tmp_path):
    result = dev_build["result"]
    again = produce_development(
        dev_build["config"], runtime=make_runtime(),
        day_objects=dev_build["day_objects"],
        matrix_path=tmp_path / "matrix.parquet",
        manifest_path=tmp_path / "manifest.json",
        generated_at="2026-07-20T09:00:00Z")  # generated_at must not matter
    assert again.write.build_id == result.write.build_id
    assert again.write.logical_row_sha256 == result.write.logical_row_sha256
    assert again.write.manifest_sha256 == result.write.manifest_sha256
    assert again.drop_counts == result.drop_counts
    assert again.row_counts == result.row_counts
    assert again.realized_threshold_schedule_sha256 == \
        result.realized_threshold_schedule_sha256
    assert again.clock_state_sha256 == result.clock_state_sha256


def test_development_data_identity_binds_the_build(dev_build):
    identity = dev_build["result"].data_identity
    development_data_identity(identity)  # fail-closed round trip
    assert identity["development_build_id"] == dev_build["result"].write.build_id
    assert identity["partition_plan_sha256"] == \
        dev_build["config"]["partition"]["sha256"]


def test_development_realized_schedule_is_causal_and_recorded(dev_build):
    sched = dev_build["result"].realized_threshold_schedule
    assert [s["day"] for s in sched] == ["2025-11-01", "2025-11-02", "2025-11-03"]
    assert sched[0]["is_warmup"] and not sched[1]["is_warmup"]
    # trailing mean over prior days only: day 2/3 thresholds are live values
    assert sched[1]["threshold"] > 0 and not sched[2]["is_warmup"]
    state = dev_build["result"].clock_state
    assert state["schema"] == "g0bn-clock-state-v1"
    assert [h["day"] for h in state["history"]] == [s["day"] for s in sched]
    # plan §A: the manifest persists the per-day VALUES plus the content hash —
    # "not a single scalar and not the hash alone" (a hash cannot recover the
    # thresholds for a rebuild/audit)
    manifest = load_manifest(dev_build["result"].write.manifest_path)
    assert manifest["bar_clock"]["threshold_schedule"] == sched
    assert manifest["bar_clock"]["threshold_schedule_hash"] == \
        dev_build["result"].realized_threshold_schedule_sha256


# ---------------------------------------------------------- targeted drop paths


def _build(tmp_path, days_spec, config=None, runtime=None, **over):
    """days_spec: iterable of (day, day_kwargs)."""
    world = SyntheticWorld()
    config = config or produce_config()
    day_objects = {}
    for day, kwargs in days_spec:
        paths, _ = write_day_objects(world, day, tmp_path, **kwargs)
        day_objects[day] = paths
    return produce_development(
        config, runtime=runtime or make_runtime(), day_objects=day_objects,
        matrix_path=tmp_path / "matrix.parquet",
        manifest_path=tmp_path / "manifest.json", generated_at=GEN_AT,
        **over), config


def test_gap_between_days_drops_first_post_gap_bar(tmp_path):
    result, _ = _build(tmp_path, [("2025-11-01", {}), ("2025-11-02", {}),
                                  ("2025-11-04", {})])
    # the prior observable read carries across the excluded 11-03 gap; the first
    # 11-04 bar's feature window escapes the segment's trusted span and must be
    # dropped (before_start), never clipped
    assert all(n >= 1 for n in result.drop_counts["before_start"].values())


def test_post_gap_windows_never_cross_the_excluded_day(tmp_path):
    """Plan §C.3 coverage integrity MID-partition: a segment-start day after an
    excluded/missing day must not emit rows whose feature window reaches into
    the gap. The day-open snapshot originates inside the excluded day, so a
    delayed first delta yields snapshot-only reads whose windows cross it —
    those bars drop as before_start (dropped, never clipped), exactly as the
    partition-start check already masks them on the partition's first day."""
    import pandas as pd

    result, _ = _build(tmp_path, [("2025-11-01", {}), ("2025-11-02", {}),
                                  ("2025-11-04",
                                   {"first_delta_offset_ns": 5_000_000_000})])
    open_1104 = day_open_ns("2025-11-04")
    frame = pd.read_parquet(tmp_path / "matrix.parquet")
    rows = frame[frame["t_event"] >= open_1104]
    assert len(rows)
    assert (rows["t_feature_start"] >= open_1104).all()
    assert all(n >= 1 for n in result.drop_counts["before_start"].values())


def test_full_scope_schedule_pin_gate(tmp_path):
    """Canonical (full included-day scope) development builds must reproduce
    `config.clock.development_schedule_sha256` exactly before anything is
    published; subset builds skip the gate (they cannot match a full-scope pin
    by construction); the pin-minting bootstrap runs with
    verify_schedule_pin=False and seals the realized hash it publishes."""
    days_spec = [(d, {}) for d in dev_included_days()]

    # 1) bootstrap mint: full scope under the placeholder pin, explicit escape
    mint_dir = tmp_path / "mint"
    mint_dir.mkdir()
    minted, _ = _build(mint_dir, days_spec, verify_schedule_pin=False)
    realized = minted.realized_threshold_schedule_sha256

    # 2) canonical run under the sealed pin: the default enforces equality
    good_dir = tmp_path / "good"
    good_dir.mkdir()
    config = produce_config(clock=make_clock(
        target_bars_per_day=TARGET_BARS_PER_DAY, time_cap_ns=TIME_CAP_NS,
        development_schedule_sha256=realized))
    result, _ = _build(good_dir, days_spec, config=config)
    assert result.realized_threshold_schedule_sha256 == realized

    # 3) a full-scope build under a non-matching pin fails closed pre-publish
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    with pytest.raises(ValueError, match="development_schedule_sha256"):
        _build(bad_dir, days_spec)
    assert not (bad_dir / "matrix.parquet").exists()
    assert not (bad_dir / "manifest.json").exists()


def test_thin_horizon_below_n_groups_fails_closed(tmp_path):
    """A non-empty build that leaves any declared horizon with fewer than
    cv.n_groups surviving rows must fail at production time: the manifest would
    declare every horizon, but the artifact is rejected downstream (empty-slice
    check / CPCV n_groups > n_samples) — successful-but-unusable."""
    target = 4  # two synthetic days yield 3 surviving rows per horizon (< 6)
    config = produce_config(clock=make_clock(target_bars_per_day=target,
                                             time_cap_ns=TIME_CAP_NS))
    runtime = make_runtime(threshold=ThresholdConfig(
        target_bars_per_day=target, window_days=3, warmup_days=1,
        seed_threshold=SEED_THRESHOLD, min_covered_fraction=0.0))
    with pytest.raises(ValueError, match="n_groups"):
        _build(tmp_path, [("2025-11-01", {}), ("2025-11-02", {})],
               config=config, runtime=runtime)
    assert not (tmp_path / "matrix.parquet").exists()
    assert not (tmp_path / "manifest.json").exists()


def test_partition_end_prefilter_is_per_horizon(tmp_path):
    # the default dev fixture scope stops at Nov 24; the December partition-end
    # days need the full-window exclusions block
    config = produce_config(exclusions=make_exclusions())
    result, _ = _build(tmp_path, [("2025-12-30", {}),
                                  ("2025-12-31", {"late_active": True})],
                       config=config)
    prefilter = result.drop_counts["prefilter"]
    assert prefilter["60s"] > 0
    # longer horizons cross the boundary earlier: strictly more 60s drops
    assert prefilter["60s"] > prefilter["2s"]
    # horizons still survive on supported earlier bars
    assert set(result.row_counts) == {"2s", "10s", "60s"}
    assert all(n > 0 for n in result.row_counts.values())


def test_coverage_gap_masks_windows_that_overrun_the_segment(tmp_path):
    result, _ = _build(tmp_path, [("2025-11-01", {}),
                                  ("2025-11-02", {"late_active": True}),
                                  ("2025-11-04", {})])
    coverage = result.drop_counts["coverage_gap"]
    assert coverage["60s"] > 0
    assert coverage["60s"] > coverage["2s"]
    # the mid-partition day boundary into 11-03 is a coverage gap, not the
    # partition-end prefilter
    assert result.drop_counts["prefilter"]["60s"] == 0


def test_one_sided_seed_book_counts_book_rejections(tmp_path):
    # day 2's seed object carries only bids and the first delta arrives at 5s:
    # bars closing before any ask level is restored reject as one_sided_book
    result, _ = _build(tmp_path, [
        ("2025-11-01", {}),
        ("2025-11-02", {"one_sided_snapshot": True,
                        "first_delta_offset_ns": 5_000_000_000})])
    assert all(n >= 1 for n in result.drop_counts["book_rejection"].values())


def test_insufficient_vol_history_counts_label_rejections(tmp_path):
    # min_returns above the trailing return count at day 2's first anchors:
    # the earliest labeled bars reject with insufficient_vol_history and later
    # bars (more path returns) still label
    # day 1 contributes ~300 trailing returns; day 2's active window adds ~5/s,
    # so 400 rejects the first ~19s of day-2 anchors and passes the rest
    runtime = make_runtime(min_returns=400)
    result, _ = _build(tmp_path, [("2025-11-01", {}), ("2025-11-02", {})],
                       runtime=runtime)
    assert all(n >= 1 for n in result.drop_counts["label_rejection"].values())
    assert all(n > 0 for n in result.row_counts.values())


def test_invalid_true_book_stretch_drops_crossing_windows(tmp_path):
    # Codex round 9 + round 18: a one-sided/crossed true book inside the
    # RESOLVED label span [t_event, t_barrier] means the resolution has no
    # covered true-mid support — those rows drop (coverage_gap). But a row
    # whose barrier touches BEFORE the stretch resolved entirely on covered
    # points and must be KEPT even though its nominal window crosses.
    import pandas as pd

    from tests.produce_fixtures import day_open_ns

    result, _ = _build(tmp_path, [("2025-11-01", {}),
                                  ("2025-11-02", {"invalid_midday": True})])
    coverage = result.drop_counts["coverage_gap"]
    # rows whose RESOLUTION reaches into/past the [30s, 32s) hole drop
    assert coverage["60s"] > 0
    assert coverage["60s"] >= coverage["10s"] >= coverage["2s"]
    # anchors AT invalid instants were already bar-level book rejections
    assert all(n >= 1 for n in result.drop_counts["book_rejection"].values())
    assert all(n > 0 for n in result.row_counts.values())
    # early-resolving 60s rows with a nominally-crossing window are KEPT: their
    # resolved span ended on covered points before the stretch began
    frame = pd.read_parquet(result.write.matrix_path)
    stretch_start = day_open_ns("2025-11-02") + 30 * 10**9
    sixty = frame[frame["horizon"] == "60s"]
    kept_crossing = sixty[(sixty["t_event"] < stretch_start)
                          & (sixty["t_event"] + 60 * 10**9 > stretch_start)
                          & (sixty["t_barrier"] <= stretch_start)]
    assert len(kept_crossing) > 0


def test_l2_objects_stream_a_constant_number_of_times(tmp_path, monkeypatch):
    # Codex round 15: label passes must share ONE segment path stream across the
    # horizon ladder — per-horizon replays would scale L2 I/O linearly with the
    # ladder. Expected passes per delta object: validity scan + dual reads +
    # one shared label path = 3, independent of the horizon count.
    import pyarrow.parquet as pq

    opens = []
    real_pf = pq.ParquetFile

    def spy(path, *args, **kwargs):
        opens.append(str(path))
        return real_pf(path, *args, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", spy)
    result, _ = _build(tmp_path, [("2025-11-01", {}), ("2025-11-02", {})])
    assert all(n > 0 for n in result.row_counts.values())
    for day in ("2025-11-01", "2025-11-02"):
        delta = str(tmp_path / f"binance_futures_l2_delta-{day}.parquet")
        assert opens.count(delta) == 3, (day, opens.count(delta))


def test_transient_same_timestamp_invalidity_is_not_a_gap(tmp_path):
    # Codex round 10: a remove/re-add at ONE origin_time is transiently invalid
    # per event, but the state AT that instant is the last event's (T2/T5
    # coalescing) — the validity scan must assess per timestamp, never emit a
    # zero-length [ts, ts) interval that falsely drops crossing windows
    import pyarrow as pa
    import pyarrow.parquet as pq

    from bars.produce import _scan_day_validity
    from tests.produce_fixtures import day_open_ns

    day = "2025-11-01"
    open_ns = day_open_ns(day)
    snap = tmp_path / "snap.parquet"
    pq.write_table(pa.table({
        "origin_time": pa.array([open_ns - 10**9] * 2, pa.int64()),
        "received_time": pa.array([open_ns - 10**9 + 1] * 2, pa.int64()),
        "seq": pa.array([1, 2], pa.int64()),
        "side": pa.array(["bid", "ask"]),
        "price": pa.array([99.0, 101.0], pa.float64()),
        "size": pa.array([5.0, 5.0], pa.float64()),
    }), snap)
    ts = open_ns + 10 * 10**9
    delta = tmp_path / "delta.parquet"
    pq.write_table(pa.table({
        "origin_time": pa.array([ts, ts], pa.int64()),
        "received_time": pa.array([ts + 1, ts + 2], pa.int64()),
        "seq": pa.array([3, 4], pa.int64()),
        "side": pa.array(["ask", "ask"]),
        "price": pa.array([101.0, 101.0], pa.float64()),
        "size": pa.array([0.0, 5.0], pa.float64()),  # remove then re-add
    }), delta)
    assert _scan_day_validity(day, snap, delta) == []


def test_missing_seed_prefix_is_an_invalid_interval(tmp_path):
    # Codex round 11: before any usable book state the day's true book is
    # MISSING — the scan must start invalid at the open and record the
    # [day_open, first_valid) prefix, or prior-day cross-midnight windows would
    # bridge a stale mid across unsupported time
    import pyarrow as pa
    import pyarrow.parquet as pq

    from bars.produce import _scan_day_validity
    from tests.produce_fixtures import day_open_ns

    day = "2025-11-01"
    open_ns = day_open_ns(day)
    empty = {"origin_time": pa.array([], pa.int64()),
             "received_time": pa.array([], pa.int64()),
             "seq": pa.array([], pa.int64()),
             "side": pa.array([], pa.string()),
             "price": pa.array([], pa.float64()),
             "size": pa.array([], pa.float64())}
    snap = tmp_path / "snap.parquet"
    pq.write_table(pa.table(empty), snap)
    first_valid = open_ns + 10 * 10**9
    delta = tmp_path / "delta.parquet"
    pq.write_table(pa.table({
        "origin_time": pa.array([first_valid] * 2, pa.int64()),
        "received_time": pa.array([first_valid + 1] * 2, pa.int64()),
        "seq": pa.array([1, 2], pa.int64()),
        "side": pa.array(["bid", "ask"]),
        "price": pa.array([99.0, 101.0], pa.float64()),
        "size": pa.array([5.0, 5.0], pa.float64()),
    }), delta)
    assert _scan_day_validity(day, snap, delta) == [(open_ns, first_valid)]
    # and a whole day with no book events at all is invalid end to end
    empty_delta = tmp_path / "empty-delta.parquet"
    pq.write_table(pa.table(empty), empty_delta)
    assert _scan_day_validity(day, snap, empty_delta) == \
        [(open_ns, open_ns + 86_400 * 10**9)]


def test_day_end_truncation_bar_is_masked(tmp_path):
    result, _ = _build(tmp_path, [("2025-11-05", {}),
                                  ("2025-11-06", {"late_trade": True})])
    assert all(n >= 1 for n in result.drop_counts["day_end_truncation"].values())


# ------------------------------------------------------------ fail-closed inputs


def test_excluded_or_out_of_window_days_are_refused(tmp_path):
    world = SyntheticWorld()
    # full-window scope: 2025-11-14 carries an explicit outcome-blind exclusion
    config = produce_config(exclusions=make_exclusions())
    paths, _ = write_day_objects(world, "2025-11-14", tmp_path)  # excluded day
    with pytest.raises(ValueError, match="included development days"):
        produce_development(
            config, runtime=make_runtime(), day_objects={"2025-11-14": paths},
            matrix_path=tmp_path / "m.parquet",
            manifest_path=tmp_path / "m.json", generated_at=GEN_AT)


def test_wrong_product_set_is_refused(tmp_path):
    world = SyntheticWorld()
    config = produce_config()
    paths, _ = write_day_objects(world, "2025-11-01", tmp_path)
    incomplete = {k: v for k, v in paths.items()
                  if k != "binance_futures_l2_delta"}
    with pytest.raises(ValueError, match="certified normalized products"):
        produce_development(
            config, runtime=make_runtime(),
            day_objects={"2025-11-01": incomplete},
            matrix_path=tmp_path / "m.parquet",
            manifest_path=tmp_path / "m.json", generated_at=GEN_AT)
    foreign = dict(paths)
    foreign["coinbase_trades"] = paths["binance_futures_trades"]
    with pytest.raises(ValueError, match="certified normalized products"):
        produce_development(
            config, runtime=make_runtime(), day_objects={"2025-11-01": foreign},
            matrix_path=tmp_path / "m.parquet",
            manifest_path=tmp_path / "m.json", generated_at=GEN_AT)


def test_glob_and_missing_paths_are_refused(tmp_path):
    world = SyntheticWorld()
    config = produce_config()
    paths, _ = write_day_objects(world, "2025-11-01", tmp_path)
    globby = dict(paths, binance_futures_trades=str(tmp_path / "*.parquet"))
    with pytest.raises(ValueError, match="glob"):
        produce_development(
            config, runtime=make_runtime(), day_objects={"2025-11-01": globby},
            matrix_path=tmp_path / "m.parquet",
            manifest_path=tmp_path / "m.json", generated_at=GEN_AT)
    gone = dict(paths, binance_futures_trades=str(tmp_path / "absent.parquet"))
    with pytest.raises(ValueError, match="existing regular file"):
        produce_development(
            config, runtime=make_runtime(), day_objects={"2025-11-01": gone},
            matrix_path=tmp_path / "m.parquet",
            manifest_path=tmp_path / "m.json", generated_at=GEN_AT)


def test_dev_manifest_source_hashes_reconcile_with_the_real_files(dev_build):
    manifest = load_manifest(dev_build["result"].write.manifest_path)
    by_day_product = {(s["day"], s["name"]): s["sha256"]
                      for s in manifest["sources"]
                      if isinstance(s, dict) and "day" in s}
    expected = {(day, product): sha
                for day, shas in dev_build["day_shas"].items()
                for product, sha in shas.items()}
    assert by_day_product == expected


def test_midnight_watermark_spillover_bars_are_masked(tmp_path):
    # a pre-midnight burst whose capture lag crosses the day end: the closing
    # bars' watermarks land past their day's midnight, so their true origin cut
    # is not constructible from the day-scoped feed — they must be masked as
    # day-boundary truncation artifacts, never labeled with a truncated P0
    result, _ = _build(tmp_path, [("2025-11-07", {}),
                                  ("2025-11-08", {"midnight_burst": True}),
                                  ("2025-11-09", {})])
    assert all(n >= 2 for n in result.drop_counts["day_end_truncation"].values())
    assert all(n > 0 for n in result.row_counts.values())


def test_boundary_spillover_reads_never_advance_feature_state(tmp_path,
                                                              monkeypatch):
    # Codex round-2 P2: a bar whose watermark crosses its day's midnight carries
    # a read computed from a knowingly truncated event basis (the day-scoped
    # feed omits next-day events observable at that t_event); it must be dropped
    # WITHOUT advancing the feature builder's prior-read state
    import bars.produce as produce_mod
    from tests.produce_fixtures import DAY_NS

    seen = []
    real_builder = produce_mod.BarFeatureBuilder

    class SpyBuilder(real_builder):
        def build(self, bar, read):
            seen.append(bar)
            return super().build(bar, read)

    monkeypatch.setattr(produce_mod, "BarFeatureBuilder", SpyBuilder)
    # late_active keeps the book fresh at the burst, so the spillover bars carry
    # VALID observable reads — the exact case that must not advance state
    _build(tmp_path, [("2025-11-07", {}),
                      ("2025-11-08", {"late_active": True,
                                      "midnight_burst": True}),
                      ("2025-11-09", {})])
    spillover = [b for b in seen
                 if b.t_event >= (b.interval_start_ns // DAY_NS + 1) * DAY_NS]
    assert spillover == [], \
        f"{len(spillover)} boundary-spillover bar(s) reached the feature builder"


def test_targets_are_log_mid_ratio_bps(tmp_path, monkeypatch):
    # Codex round 5: the protocol config pins labels.return_formula ==
    # log_mid_ratio_bps_v1, while data.labels emits simple mid-ratio bps; the
    # published y_fwd_bps must be the exact log conversion of T5's return
    import math

    import pandas as pd

    import bars.produce as produce_mod
    from data.labels import LabelRow

    captured = {}
    real_tbl = produce_mod.triple_barrier_labels

    def spy(path, anchors, **kwargs):
        for out in real_tbl(path, anchors, **kwargs):
            if isinstance(out, LabelRow):
                captured[(out.t_event, out.horizon)] = out.y_fwd_bps
            yield out

    monkeypatch.setattr(produce_mod, "triple_barrier_labels", spy)
    result, _ = _build(tmp_path, [("2025-11-01", {}), ("2025-11-02", {})])
    frame = pd.read_parquet(result.write.matrix_path)
    assert (frame["y_fwd_bps"] != 0).any()
    for _, row in frame.iterrows():
        raw = captured[(row["t_event"], row["horizon"])]
        assert row["y_fwd_bps"] == pytest.approx(
            1e4 * math.log1p(raw / 1e4), rel=1e-14, abs=1e-14)


def test_off_day_delta_rows_fail_closed(tmp_path):
    # Codex round 5: trades and snapshots are day-bounded, so a delta row
    # outside its object's declared UTC day must fail closed before it can
    # advance any book fold
    import pyarrow as pa
    import pyarrow.parquet as pq

    from tests.produce_fixtures import DAY_NS, day_open_ns

    world = SyntheticWorld()
    config = produce_config()
    paths, _ = write_day_objects(world, "2025-11-01", tmp_path)
    delta_path = paths["binance_futures_l2_delta"]
    table = pq.read_table(delta_path)
    off_day = day_open_ns("2025-11-01") + DAY_NS + 5
    extra = pa.table({
        "origin_time": pa.array([off_day], pa.int64()),
        "received_time": pa.array([off_day + 1], pa.int64()),
        "seq": pa.array([10**9], pa.int64()),
        "side": pa.array(["bid"]),
        "price": pa.array([99.99], pa.float64()),
        "size": pa.array([5.0], pa.float64()),
    })
    pq.write_table(pa.concat_tables([table.select(extra.column_names), extra]),
                   delta_path)
    with pytest.raises(ValueError, match="outside its declared day"):
        produce_development(
            config, runtime=make_runtime(), day_objects={"2025-11-01": paths},
            matrix_path=tmp_path / "m.parquet",
            manifest_path=tmp_path / "m.json", generated_at=GEN_AT)


def test_malformed_rows_fail_closed_at_the_reader_boundary(tmp_path):
    # Codex round 6: the lazily-consumed label-path fold can reach L2 rows that
    # dual_book_reads never validates (its per-event checks cover only the
    # consumed decision prefix), so the READER itself must apply the same
    # side/price/size/received-time validation before any fold sees an event
    import pyarrow as pa
    import pyarrow.parquet as pq

    t0 = 1_800_000_000_000_000_000
    cases = (
        ({"side": "foo"}, "side"),
        ({"price": float("nan")}, "price"),
        ({"size": -1.0}, "size"),
        ({"received_time": t0 - 1}, "received_time"),
    )
    for i, (override, match) in enumerate(cases):
        row = {"origin_time": t0, "received_time": t0 + 1, "seq": 1,
               "side": "bid", "price": 99.99, "size": 5.0}
        row.update(override)
        path = tmp_path / f"bad-{i}.parquet"
        pq.write_table(pa.table({
            "origin_time": pa.array([row["origin_time"]], pa.int64()),
            "received_time": pa.array([row["received_time"]], pa.int64()),
            "seq": pa.array([row["seq"]], pa.int64()),
            "side": pa.array([row["side"]]),
            "price": pa.array([row["price"]], pa.float64()),
            "size": pa.array([row["size"]], pa.float64()),
        }), path)
        with pytest.raises(ValueError, match=match):
            list(iter_normalized_book_events(path))


def test_unpulled_l2_tail_is_still_validated(tmp_path):
    # Codex round 8: rows past the last decision's consumed prefix AND past the
    # last label window are never pulled by any lazy fold — a sealed object
    # with a malformed tail row must still fail the build (the full object is
    # drained through the validating reader before publication)
    import pyarrow as pa
    import pyarrow.parquet as pq

    from tests.produce_fixtures import day_open_ns

    world = SyntheticWorld()
    config = produce_config()
    day_objects = {}
    for day in ("2025-11-01", "2025-11-02"):
        paths, _ = write_day_objects(world, day, tmp_path)
        day_objects[day] = paths
    delta_path = day_objects["2025-11-02"]["binance_futures_l2_delta"]
    table = pq.read_table(delta_path)
    good = day_open_ns("2025-11-02") + 79_200 * 10**9  # 22:00: absorbs the
    late = day_open_ns("2025-11-02") + 84_600 * 10**9  # sampler's one-event
    bad = pa.table({                                   # lookahead; 23:30 is
        "origin_time": pa.array([good, late], pa.int64()),  # then truly
        "received_time": pa.array([good + 1, late + 1], pa.int64()),  # unpulled
        "seq": pa.array([10**9, 10**9 + 1], pa.int64()),
        "side": pa.array(["bid", "foo"]),
        "price": pa.array([99.99, 99.99], pa.float64()),
        "size": pa.array([5.0, 5.0], pa.float64()),
    })
    pq.write_table(pa.concat_tables([table.select(bad.column_names), bad]),
                   delta_path)
    with pytest.raises(ValueError, match="side"):
        produce_development(
            config, runtime=make_runtime(), day_objects=day_objects,
            matrix_path=tmp_path / "m.parquet",
            manifest_path=tmp_path / "m.json", generated_at=GEN_AT)


def test_pinned_open_rejects_non_regular_files(tmp_path):
    # Codex round 5: the fd-level gate must reject a FIFO/device swapped in
    # after the path checks (O_NOFOLLOW alone only stops symlinks, and a FIFO
    # open would otherwise block)
    import os

    from bars.produce import _open_pinned_fd

    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular file"):
        _open_pinned_fd(str(fifo))


def test_post_open_snapshot_origin_fails_closed(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from tests.produce_fixtures import day_open_ns

    world = SyntheticWorld()
    config = produce_config()
    paths, _ = write_day_objects(world, "2025-11-01", tmp_path)
    bad_origin = day_open_ns("2025-11-01") + 10
    pq.write_table(pa.table({
        "origin_time": pa.array([bad_origin], pa.int64()),
        "received_time": pa.array([bad_origin + 1], pa.int64()),
        "seq": pa.array([1], pa.int64()),
        "side": pa.array(["bid"]),
        "price": pa.array([99.99], pa.float64()),
        "size": pa.array([5.0], pa.float64()),
    }), paths["binance_futures_l2_snapshot"])
    with pytest.raises(ValueError, match="after the day open"):
        produce_development(
            config, runtime=make_runtime(), day_objects={"2025-11-01": paths},
            matrix_path=tmp_path / "m.parquet",
            manifest_path=tmp_path / "m.json", generated_at=GEN_AT)


def test_empty_build_is_never_published(tmp_path):
    # a single fresh dev day is entirely warm-up: every row drops and the build
    # must refuse to publish an empty matrix
    with pytest.raises(ValueError, match="no surviving rows"):
        _build(tmp_path, [("2025-11-01", {})])
    assert not (tmp_path / "matrix.parquet").exists()


def test_l2_objects_stream_batchwise(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    real_pf = pq.ParquetFile

    class GuardedParquetFile:
        """Full-table .read() is forbidden for L2 objects: the bounded-memory
        contract streams book events batchwise (trades may materialize a day)."""

        def __init__(self, path, *args, **kwargs):
            self._l2 = ("l2_snapshot" in str(path)) or ("l2_delta" in str(path))
            self._pf = real_pf(path, *args, **kwargs)

        def read(self, *args, **kwargs):
            if self._l2:
                raise AssertionError(
                    "L2 book objects must stream via iter_batches, never a "
                    "full-day read()")
            return self._pf.read(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._pf, name)

    monkeypatch.setattr(pq, "ParquetFile", GuardedParquetFile)
    result, _ = _build(tmp_path, [("2025-11-01", {}), ("2025-11-02", {})])
    assert all(n > 0 for n in result.row_counts.values())


def test_foreign_clock_rule_identities_fail_closed(tmp_path):
    world = SyntheticWorld()
    paths, _ = write_day_objects(world, "2025-11-01", tmp_path)
    for override, match in (
            ({"adaptive_threshold_update_rule": "median_of_medians_v9"},
             "adaptive_threshold_update_rule"),
            ({"coverage_normalization": "no_normalization_v3"},
             "coverage_normalization")):
        config = produce_config(clock=make_clock(
            target_bars_per_day=TARGET_BARS_PER_DAY, time_cap_ns=TIME_CAP_NS,
            **override))
        with pytest.raises(ValueError, match=match):
            produce_development(
                config, runtime=make_runtime(),
                day_objects={"2025-11-01": paths},
                matrix_path=tmp_path / "m.parquet",
                manifest_path=tmp_path / "m.json", generated_at=GEN_AT)


def test_day_bridging_lookback_cap_fails_closed(tmp_path):
    world = SyntheticWorld()
    paths, _ = write_day_objects(world, "2025-11-01", tmp_path)
    config = produce_config(
        producer=make_producer(lookback_cap_ns=2 * 86_400 * 10**9),
        features=make_features(max_lookback_ns=2 * 86_400 * 10**9),
        cv=runtime_cv(embargo_ns=2 * 86_400 * 10**9))
    with pytest.raises(ValueError, match="under one UTC day"):
        produce_development(
            config, runtime=make_runtime(), day_objects={"2025-11-01": paths},
            matrix_path=tmp_path / "m.parquet",
            manifest_path=tmp_path / "m.json", generated_at=GEN_AT)


def test_degenerate_barrier_runtime_is_rejected_at_the_boundary(tmp_path):
    # Codex P1: min_returns=0 / negative vol_floor_bps must fail the eager
    # runtime validation, never surface later inside the streaming build
    world = SyntheticWorld()
    config = produce_config()
    paths, _ = write_day_objects(world, "2025-11-01", tmp_path)
    for bad, match in ((make_runtime(min_returns=0), "min_returns"),
                       (make_runtime(vol_floor_bps=-1.0), "vol_floor_bps"),
                       (make_runtime(vol_floor_bps=float("nan")), "vol_floor_bps"),
                       # a float top_k passes FeatureConfig's >=1 check but
                       # would first explode inside heapq.nlargest post-burn
                       (make_runtime(top_k=3.0), "top_k"),
                       (make_runtime(top_k=True), "top_k"),
                       # JSON/YAML typo classes: bools pass numeric validators
                       # as 1/0 and fractional day/count fields pass comparisons
                       (make_runtime(tick_size=True), "tick_size"),
                       (make_runtime(threshold=ThresholdConfig(
                           target_bars_per_day=TARGET_BARS_PER_DAY,
                           window_days=3, warmup_days=1,
                           seed_threshold=True, min_covered_fraction=0.0)),
                        "seed_threshold"),
                       (make_runtime(threshold=ThresholdConfig(
                           target_bars_per_day=TARGET_BARS_PER_DAY,
                           window_days=3.5, warmup_days=1,
                           seed_threshold=SEED_THRESHOLD,
                           min_covered_fraction=0.0)), "window_days"),
                       (make_runtime(threshold=ThresholdConfig(
                           target_bars_per_day=True, window_days=3,
                           warmup_days=1, seed_threshold=SEED_THRESHOLD,
                           min_covered_fraction=0.0)), "target_bars_per_day")):
        with pytest.raises(ValueError, match=match):
            produce_development(
                config, runtime=bad, day_objects={"2025-11-01": paths},
                matrix_path=tmp_path / "m.parquet",
                manifest_path=tmp_path / "m.json", generated_at=GEN_AT)


def test_cost_source_must_match_the_certified_normalized_identity(tmp_path):
    # Codex round 19: the assumption's source must bind the config-pinned
    # normalized contract identity (source_certification.normalized_schema_
    # version), not echo itself — a mistyped/copied source fails closed
    from tests.g0bn_protocol_fixtures import make_costs

    world = SyntheticWorld()
    paths, _ = write_day_objects(world, "2025-11-01", tmp_path)
    foreign = make_costs()
    foreign["cost_assumption"] = dict(foreign["cost_assumption"],
                                      source="coinbase/normalized_v9")
    config = produce_config(costs=foreign)
    with pytest.raises(ValueError, match="cost assumption identity mismatch"):
        produce_development(
            config, runtime=make_runtime(), day_objects={"2025-11-01": paths},
            matrix_path=tmp_path / "m.parquet",
            manifest_path=tmp_path / "m.json", generated_at=GEN_AT)


def test_runtime_must_match_the_config_clock_pin(tmp_path):
    world = SyntheticWorld()
    config = produce_config()
    paths, _ = write_day_objects(world, "2025-11-01", tmp_path)
    bad = make_runtime(threshold=ThresholdConfig(
        target_bars_per_day=TARGET_BARS_PER_DAY + 1, window_days=3,
        warmup_days=1, seed_threshold=SEED_THRESHOLD, min_covered_fraction=0.0))
    with pytest.raises(ValueError, match="target_bars_per_day"):
        produce_development(
            config, runtime=bad, day_objects={"2025-11-01": paths},
            matrix_path=tmp_path / "m.parquet",
            manifest_path=tmp_path / "m.json", generated_at=GEN_AT)
