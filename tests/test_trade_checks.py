"""Offline synthetic tests for the pure trade-feed validation checks
(`ingest/trade_checks.py`, docs/data.md §5b / §10 "trade validation breadth", plan
docs/superpowers/plans/2026-07-02-trade-validation-breadth-plan.md §9 Phase-1a).

The module is source-agnostic and pure — it validates a *normalized* trade frame (the loaded/renamed
Crypto Lake or CoinAPI schema) with pandas/numpy only, no lakeapi/boto3 import. Every check here runs
on in-memory synthetic frames, so CI never touches a vendor (mirrors tests/test_quality_map.py). The
CLI-guard case (§9 case 13) belongs to the Phase-1b Lake-wrapper branch and is not tested here."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from ingest import trade_checks as tc

# lakeapi returns origin_time/received_time as tz-NAIVE datetime64[ns] (§4), and the validator's null
# sentinel `< pd.Timestamp("2015-01-01")` is tz-naive too. Keep every synthetic timestamp naive (no
# "Z"/tz on `start`) — a tz-aware column vs. the naive cutoff raises "Cannot compare tz-naive and
# tz-aware", and assigning the naive SENTINEL into a tz-aware column coerces it to object.
SENTINEL = pd.Timestamp("1970-01-01")          # < 2015-01-01 → treated as null (§4/§5), tz-naive


def _trades_df(n=1000, start="2025-06-01T00:00:00", step_ms=80, *, presorted=True,
               full_day=False, null_origin=0.0, null_received=0.0, dup_ids=0,
               bad_price=False, bad_size=False, spike_price=False, empty_hours=()):
    """Deterministic synthetic Crypto Lake `trades` frame (loaded/renamed columns).

    full_day=True spreads the n rows evenly across [00:00, 24:00) (step = 86_400_000 // n ms) so the
    24-hour missing/sparse-hour metric is exercisable; otherwise rows are step_ms apart from `start`.
    null_origin / null_received are FRACTIONS of leading rows set to the 1970 sentinel; the same
    leading rows overlap, so null_origin>0 with null_received>0 makes those rows unrecoverable.
    dup_ids is the number of EXTRA duplicate trade_ids created (first dup_ids+1 rows share one id).
    spike_price sets one middle row to an 11× price (660000 among constant 60000, just past the
    median×10 band) — an isolated corrupt print p99 abs-return cannot see but price_max_abs_ret /
    the robust band catch."""
    base = pd.Timestamp(start)
    step = (86_400_000 // n) if full_day else step_ms
    origin = base + pd.to_timedelta(np.arange(n) * step, unit="ms")
    if not presorted:                          # Coinbase shape: shuffle the file order
        origin = origin[np.random.RandomState(0).permutation(n)]
    df = pd.DataFrame({
        "origin_time": origin,
        "received_time": origin + pd.to_timedelta(160, unit="ms"),
        "price": np.full(n, 60000.0),
        "quantity": np.full(n, 0.01),
        "side": np.where(np.arange(n) % 2, "buy", "sell"),
        "trade_id": np.arange(n, dtype="int64"),
    })
    if null_origin:   df.loc[df.index[: int(n * null_origin)], "origin_time"] = SENTINEL
    if null_received: df.loc[df.index[: int(n * null_received)], "received_time"] = SENTINEL
    if dup_ids:       df.loc[df.index[: dup_ids + 1], "trade_id"] = df["trade_id"].iloc[0]
    if bad_price:     df.loc[df.index[0], "price"] = 0.0
    if bad_size:      df.loc[df.index[0], "quantity"] = -1.0
    if spike_price:   df.loc[df.index[n // 2], "price"] = 660000.0   # one 11× print among constants
    for h in empty_hours:
        df = df[df["origin_time"].dt.hour != h]
    return df.reset_index(drop=True)


def _clean_full_day(n=2400, start="2025-06-01T00:00:00"):
    """A full-day frame with ~n/24 rows in every UTC hour → classifies `pass` (no coverage fail).
    `start` sets the frame's UTC date so it can be matched to the requested `day` (§1 existence)."""
    return _trades_df(n=n, full_day=True, start=start)


# --------------------------------------------------------------------------- 1. required ts field
def test_missing_timestamp_field_fails_clearly():
    df = _clean_full_day().drop(columns=["origin_time"])   # no origin_time and no raw `timestamp`
    res = tc.validate_trade_frame(df, "coinbase", "2025-06-01")
    assert res["status"] == tc.FAIL
    assert tc.ORIGIN_TIME_COLUMN_MISSING in res["reason_codes"]


def test_wrong_day_partition_is_rejected():
    # §1 existence: a frame whose engine clock is for a DIFFERENT UTC day (a mislabeled partition /
    # wrong file) must not pass — even a complete 24 h stream — so a loader/normalizer supplying the
    # wrong partition can't silently clear the gate with zero trades from the requested date.
    df = _clean_full_day(start="2025-06-01T00:00:00")
    off = tc.validate_trade_frame(df, "coinbase", "2025-06-02")   # frame is 06-01, requested 06-02
    assert off["status"] == tc.FAIL and tc.WRONG_DAY_PARTITION in off["reason_codes"]
    assert off["metrics"]["off_day_frac"] == 1.0
    on = tc.validate_trade_frame(df, "coinbase", "2025-06-01")    # matching day passes
    assert on["status"] == tc.PASS and on["metrics"]["off_day_frac"] == 0.0


def test_missing_required_trade_columns_fail_cleanly():
    # §2 required columns: a present frame that omits price/quantity must return a per-day FAIL record
    # (schema drift surfaced in the JSON report), NOT raise KeyError and abort the validation run.
    for col in ("price", "quantity"):
        res = tc.validate_trade_frame(_clean_full_day().drop(columns=[col]), "coinbase", "2025-06-01")
        assert res["status"] == tc.FAIL
        assert tc.REQUIRED_COLUMN_MISSING in res["reason_codes"]
        assert f"{tc.REQUIRED_COLUMN_MISSING}:{col}" in res["reason_codes"]


# --------------------------------------------------------------------------- 2. origin_time fallback
def test_subthreshold_null_origin_falls_back_to_received_and_warns():
    # 0.5% ≤ origin_time_null_max: every null-origin row takes received_time into the engine clock;
    # the clock stays monotonic and the day is USABLE but surfaced.
    df = _trades_df(n=2400, full_day=True, null_origin=0.005)
    res = tc.validate_trade_frame(df, "coinbase", "2025-06-01")
    assert res["status"] == tc.WARN
    assert tc.RECEIVED_TIME_FALLBACK_USED in res["reason_codes"]
    assert res["metrics"]["used_received_time_fallback"] is True
    assert res["metrics"]["monotonic_after_sort"] is True


def test_superthreshold_null_origin_fails_even_though_substituted():
    # 2% > 1%: the per-row substitution still happens, but too much of the day is off exchange time.
    df = _trades_df(n=2400, full_day=True, null_origin=0.02)
    res = tc.validate_trade_frame(df, "coinbase", "2025-06-01")
    assert res["status"] == tc.FAIL
    assert tc.ORIGIN_TIME_NULL_FRACTION_HIGH in res["reason_codes"]


def test_unrecoverable_clock_fails_at_any_fraction():
    # The same leading rows are null in BOTH clocks → those rows have no resolvable engine time.
    df = _trades_df(n=2400, full_day=True, null_origin=0.005, null_received=0.005)
    res = tc.validate_trade_frame(df, "coinbase", "2025-06-01")
    assert res["status"] == tc.FAIL
    assert tc.RECEIVED_TIME_FALLBACK_UNAVAILABLE in res["reason_codes"]
    # the unresolvable clock is also surfaced as the distinct non-monotonic reason a consumer may
    # key on (both belong on the record per §9 case 3 / §6)
    assert tc.NONMONOTONIC_AFTER_SORT in res["reason_codes"]


# --------------------------------------------------------------------------- 3. monotonic after sort
def test_unsorted_frame_is_repaired_by_stable_sort():
    # The Coinbase shape: the file is NOT origin_time-ordered, but the stable sort makes the engine
    # clock non-decreasing. (Isolated pure check, so the short frame's hour coverage is irrelevant.)
    df = _trades_df(n=1000, presorted=False)
    assert tc.monotonic_after_sort(df) is True
    assert tc.was_presorted(df) is False


def test_unresolvable_clock_is_not_monotonic_after_sort():
    # A NaT/sentinel the sort cannot repair (both clocks null) is treated as invalid — it must NOT
    # read as trivially monotonic (the 1970 sentinels would otherwise sort to the front).
    df = _trades_df(n=1000, null_origin=0.01, null_received=0.01)
    assert tc.monotonic_after_sort(df) is False


# --------------------------------------------------------------------------- 4. duplicate trade ids
def test_duplicate_trade_ids_counted():
    df = _trades_df(n=1000, dup_ids=5)                      # first 6 rows share one id → 5 extra dups
    assert tc.dup_trade_ids(df)["dup_trade_id_count"] == 5
    res = tc.validate_trade_frame(_trades_df(n=2400, full_day=True, dup_ids=5),
                                  "coinbase", "2025-06-01")
    assert res["status"] == tc.WARN
    assert tc.DUPLICATE_TRADE_ID in res["reason_codes"]


# --------------------------------------------------------------------------- 5. invalid price / size
def test_nonpositive_price_fails():
    df = _trades_df(n=1000, bad_price=True)
    assert tc.price_checks(df)["price_min"] <= 0.0
    res = tc.validate_trade_frame(_trades_df(n=2400, full_day=True, bad_price=True),
                                  "coinbase", "2025-06-01")
    assert res["status"] == tc.FAIL
    assert tc.PRICE_OUT_OF_RANGE in res["reason_codes"]


def test_nan_price_is_rejected():
    # §4 row 9 / §6: "any price <= 0 OR NaN" must fail. An isolated NaN among positive prices slips
    # past np.nanmin/np.nansum and the robust band, so without an explicit non-finite count it would
    # read `pass` and corrupt the notional bar clock.
    df = _clean_full_day()
    df.loc[10, "price"] = np.nan
    assert tc.price_checks(df)["price_nonfinite_count"] == 1
    res = tc.validate_trade_frame(df, "coinbase", "2025-06-01")
    assert res["status"] == tc.FAIL and tc.PRICE_OUT_OF_RANGE in res["reason_codes"]


def test_negative_and_zero_size_fail():
    neg = tc.size_checks(_trades_df(n=1000, bad_size=True))
    assert neg["size_neg_frac"] > 0.0
    zero_df = _trades_df(n=1000)
    zero_df.loc[0, "quantity"] = 0.0
    assert tc.size_checks(zero_df)["size_zero_frac"] > 0.0
    res = tc.validate_trade_frame(_trades_df(n=2400, full_day=True, bad_size=True),
                                  "coinbase", "2025-06-01")
    assert res["status"] == tc.FAIL
    assert tc.SIZE_NONPOSITIVE in res["reason_codes"]


def test_out_of_band_size_blocks_but_large_plausible_size_warns():
    hard = _trades_df(n=2400, full_day=True)
    hard.loc[0, "quantity"] = 6000.0                       # > size_hard_max_btc → blocking
    res_hard = tc.validate_trade_frame(hard, "coinbase", "2025-06-01")
    assert res_hard["status"] == tc.FAIL
    assert tc.SIZE_OUT_OF_BAND in res_hard["reason_codes"]
    assert tc.SIZE_OUT_OF_RANGE not in res_hard["reason_codes"]   # the hard band takes precedence
    big = _trades_df(n=2400, full_day=True)
    big.loc[0, "quantity"] = 800.0                         # > size_max_btc, < hard ceiling → warn
    res_big = tc.validate_trade_frame(big, "coinbase", "2025-06-01")
    assert res_big["status"] == tc.WARN
    assert tc.SIZE_OUT_OF_RANGE in res_big["reason_codes"]


# ------------------------------------------------------------------------ 5b. isolated price spike
def test_isolated_price_spike_blocks_but_broad_churn_only_warns():
    # One 11× print: p99 abs-return can't see the two spike diffs among 1000, but price_max_abs_ret
    # and the robust median×10 band do → price_spike blocks (it would poison the notional clock).
    spike = tc.price_checks(_trades_df(n=1000, spike_price=True))
    assert spike["price_p99_abs_ret"] < 0.10                # p99 misses the lone outlier
    assert spike["price_out_of_band_count"] == 1
    res_spike = tc.validate_trade_frame(_trades_df(n=2400, full_day=True, spike_price=True),
                                        "coinbase", "2025-06-01")
    assert res_spike["status"] == tc.FAIL
    assert tc.PRICE_SPIKE in res_spike["reason_codes"]

    # A broad volatile regime (p99 > price_jump_warn, no out-of-band print) is REAL churn → warn.
    churn = _trades_df(n=2400, full_day=True)
    churn["price"] = np.where(np.arange(len(churn)) % 2, 68000.0, 60000.0)   # ~13% each tick
    metrics = tc.price_checks(churn)
    assert metrics["price_p99_abs_ret"] > 0.10 and metrics["price_out_of_band_count"] == 0
    res_churn = tc.validate_trade_frame(churn, "coinbase", "2025-06-01")
    assert res_churn["status"] == tc.WARN
    assert tc.PRICE_JUMP_EXCESS in res_churn["reason_codes"]
    assert tc.PRICE_SPIKE not in res_churn["reason_codes"]


# --------------------------------------------------------------------------- 6. sparse / missing hour
def test_one_missing_hour_warns_but_two_escalate_to_blocking():
    one = tc.validate_trade_frame(_trades_df(n=2400, full_day=True, empty_hours=(4,)),
                                  "coinbase", "2025-06-01")
    assert one["metrics"]["missing_hour_count"] == 1
    assert one["status"] == tc.WARN and tc.MISSING_HOUR in one["reason_codes"]

    two_df = _trades_df(n=2400, full_day=True, empty_hours=(3, 4))
    two = tc.validate_trade_frame(two_df, "coinbase", "2025-06-01")     # required non-fill day
    assert two["status"] == tc.FAIL and tc.MISSING_HOURS_EXCESS in two["reason_codes"]

    # ...the SAME frame routes, not fails, on a calendar fill / excluded day (neither blocks).
    fill = tc.validate_trade_frame(two_df, "coinbase", "2025-06-01",
                                   calendar_state={"route": tc.ROUTE_COINAPI_FILL})
    assert fill["status"] == tc.COINAPI_FILL
    excl = tc.validate_trade_frame(two_df, "coinbase", "2025-06-01",
                                   calendar_state={"route": tc.ROUTE_EXCLUDED})
    assert excl["status"] == tc.EXCLUDED


def test_hour_coverage_is_scoped_to_the_requested_day():
    # §4 row 13: coverage must count hours WITHIN [day, day+1), not hour-of-day across the whole
    # frame. A frame spilling into the next UTC day must not let the neighbor's hour-00 rows mask a
    # real missing first hour of the requested day (off_day_frac ~1/24 stays under threshold).
    df = _clean_full_day(start="2025-06-01T00:00:00")
    df["origin_time"] = df["origin_time"] + pd.Timedelta(hours=1)   # 01:00 Jun-1 .. 00:59 Jun-2
    df["received_time"] = df["received_time"] + pd.Timedelta(hours=1)
    res = tc.validate_trade_frame(df, "coinbase", "2025-06-01")
    assert res["metrics"]["missing_hour_count"] == 1               # hour 00 of Jun-1 is genuinely empty
    assert res["status"] == tc.WARN and tc.MISSING_HOUR in res["reason_codes"]


def test_sparse_hour_warns_at_any_count():
    df = _trades_df(n=2400, full_day=True)
    # Thin hour 5 below sparse_hour_min_rows: keep only 30 of its rows (a non-empty hour is plausibly
    # quiet, so any count below the floor is warn, never fail).
    h5 = df[df["origin_time"].dt.hour == 5]
    keep = pd.concat([df[df["origin_time"].dt.hour != 5], h5.iloc[:30]]).reset_index(drop=True)
    cov = tc.hour_coverage(keep)
    assert cov["sparse_hour_count"] >= 1 and cov["missing_hour_count"] == 0
    res = tc.validate_trade_frame(keep, "coinbase", "2025-06-01")
    assert res["status"] == tc.WARN and tc.SPARSE_HOUR in res["reason_codes"]


def test_hour_coverage_uses_post_fallback_engine_clock():
    # Substituted rows land in their real hour, not 1970 (the P2 clock fix): a sub-threshold fallback
    # frame keeps full 24-hour coverage.
    df = _trades_df(n=2400, full_day=True, null_origin=0.004)
    assert tc.hour_coverage(df)["missing_hour_count"] == 0


# --------------------------------------------------------------------------- 7. duplicate-ts cluster
def test_duplicate_timestamp_cluster_metrics():
    df = _trades_df(n=1000)
    df.loc[df.index[:60], "origin_time"] = df["origin_time"].iloc[0]    # a 60-deep same-ns cluster
    m = tc.dup_timestamp_clusters(df)
    assert m["dup_ts_max_cluster"] >= 60 and m["dup_ts_cluster_count"] >= 1
    # a small burst does not trip the warn threshold
    small = _trades_df(n=1000)
    small.loc[small.index[:3], "origin_time"] = small["origin_time"].iloc[0]
    assert tc.dup_timestamp_clusters(small)["dup_ts_max_cluster"] <= tc.THRESHOLDS.dup_ts_cluster_warn


# --------------------------------------------------------------------------- 8. inter-arrival gap
def test_interarrival_gap_summary():
    df = _trades_df(n=1000)
    df.loc[df.index[500:], "origin_time"] += pd.Timedelta(seconds=200)
    df.loc[df.index[500:], "received_time"] += pd.Timedelta(seconds=200)
    ia = tc.interarrival(df)
    assert ia["interarrival_max_s"] > 199.0
    assert ia["interarrival_median_s"] < 1.0               # the bulk is still ~80 ms apart


# --------------------------------------------------------------------------- 9. calendar + gate
def _calendar_dict():
    return {
        "usable_days": ["2025-06-01", "2024-08-06", "2025-02-02"],
        "coinbase_fill_days": {"2024-08-06": {"book": True, "trades": True}},
        "excluded_days_by_reason": {"2025-02-02": ["missing:binF_book"]},
        "lake_all_days": ["2025-06-01"],
    }


def _write_calendar(tmp_path, cal):
    p = tmp_path / "usable_calendar.json"
    p.write_text(json.dumps(cal))
    return str(p)


def test_calendar_routes_fill_and_excluded_days(tmp_path):
    cal = tc.load_usable_calendar(_write_calendar(tmp_path, _calendar_dict()))
    fill_state = tc.calendar_state(cal, "2024-08-06", "coinbase")
    fill = tc.validate_trade_frame(None, "coinbase", "2024-08-06", calendar_state=fill_state)
    assert fill["status"] == tc.COINAPI_FILL
    excl_state = tc.calendar_state(cal, "2025-02-02", "coinbase")
    excl = tc.validate_trade_frame(None, "coinbase", "2025-02-02", calendar_state=excl_state)
    assert excl["status"] == tc.EXCLUDED
    # a Binance venue is unaffected by a Coinbase trades-fill day → required
    assert tc.calendar_state(cal, "2024-08-06", "binance_spot")["route"] == tc.ROUTE_REQUIRED


def test_gate_booleans_reflect_fails_and_deferred_fills():
    clean = tc.validate_trade_frame(_clean_full_day(), "binance_spot", "2025-06-01")
    deferred = tc.validate_trade_frame(None, "coinbase", "2024-08-06",
                                       calendar_state={"route": tc.ROUTE_COINAPI_FILL})
    rep = tc.build_report([clean, deferred], meta={"generated_utc": "2026-07-02T00:00:00+00:00"})
    gate = rep["summary"]["gate"]
    assert gate["lake_required_pass"] is True         # no required-day fail...
    assert gate["bars_ready"] is False                # ...but a deferred fill blocks readiness
    assert {"day": "2024-08-06", "venue": "coinbase"} in gate["coinapi_fill_deferred"]

    # no fill days → bars_ready holds
    rep2 = tc.build_report([clean], meta={"generated_utc": "2026-07-02T00:00:00+00:00"})
    assert rep2["summary"]["gate"]["bars_ready"] is True

    # a required-day fail flips lake_required_pass and lands in blocking_failures
    bad = tc.validate_trade_frame(_trades_df(n=2400, full_day=True, bad_size=True),
                                  "coinbase", "2025-06-01")
    rep3 = tc.build_report([clean, bad], meta={"generated_utc": "2026-07-02T00:00:00+00:00"})
    g3 = rep3["summary"]["gate"]
    assert g3["lake_required_pass"] is False
    assert any(f["day"] == "2025-06-01" and f["venue"] == "coinbase"
               for f in g3["blocking_failures"])


def test_coinapi_source_fill_day_is_validated_not_re_deferred():
    # §8: Phase 3b runs the SAME validate_trade_frame on the normalized CoinAPI trade file for a fill
    # day with vendor_source="coinapi"; a pass/warn is what CLEARS coinapi_fill_deferred. So a clean
    # CoinAPI fill must classify on its own metrics, NOT route straight back to coinapi_fill (which
    # would leave bars_ready permanently false for every fill-day span).
    fill = {"route": tc.ROUTE_COINAPI_FILL}
    clean = tc.validate_trade_frame(_clean_full_day(start="2024-08-06T00:00:00"), "coinbase",
                                    "2024-08-06", calendar_state=fill, vendor_source="coinapi")
    assert clean["status"] == tc.PASS and clean["vendor_source"] == "coinapi"
    # a bad CoinAPI fill still fails on its merits (surfaced, not silently deferred)
    bad = tc.validate_trade_frame(
        _trades_df(n=2400, full_day=True, start="2024-08-06T00:00:00", bad_size=True),
        "coinbase", "2024-08-06", calendar_state=fill, vendor_source="coinapi")
    assert bad["status"] == tc.FAIL
    # the Lake-side path on the same fill day still defers (unchanged: the missing Lake side)
    assert tc.validate_trade_frame(None, "coinbase", "2024-08-06",
                                   calendar_state=fill)["status"] == tc.COINAPI_FILL


# --------------------------------------------------------------------------- 10. report JSON stability
def test_report_is_strict_json_and_byte_deterministic(tmp_path):
    records = [
        tc.validate_trade_frame(_clean_full_day(), "binance_perp", "2025-06-01"),
        tc.validate_trade_frame(_clean_full_day(), "binance_spot", "2025-06-01"),
        tc.validate_trade_frame(None, "coinbase", "2024-08-06",
                                calendar_state={"route": tc.ROUTE_COINAPI_FILL}),
    ]
    meta = {"generated_utc": "2026-07-02T00:00:00+00:00",
            "thresholds": tc.THRESHOLDS.as_dict()}
    rep = tc.build_report(records, meta=meta)
    out1 = tmp_path / "trade_feed_validation.json"
    out2 = tmp_path / "trade_feed_validation_2.json"
    tc.write_report(rep, str(out1))
    tc.write_report(tc.build_report(records, meta=meta), str(out2))
    txt = out1.read_text()
    assert "NaN" not in txt and "Infinity" not in txt      # strict JSON, allow_nan=False
    assert json.loads(txt) == json.loads(out2.read_text())  # round-trips...
    assert txt == out2.read_text()                          # ...and is byte-for-byte deterministic
    # every status/venue present in the summary even at zero count (stable schema)
    counts = rep["summary"]["counts"]
    for s in tc.STATUSES:
        assert s in counts
    for v in tc.VENUES:
        assert v in rep["summary"]["by_venue"]


# --------------------------------------------------------------------------- 11. malformed calendar
def test_malformed_calendar_entry_raises(tmp_path):
    bad = _calendar_dict()
    bad["coinbase_fill_days"] = {"2024-08-06": True}       # not a {book, trades} dict
    cal = tc.load_usable_calendar(_write_calendar(tmp_path, bad))
    with pytest.raises(ValueError, match="coinbase_fill_days"):
        tc.calendar_state(cal, "2024-08-06", "coinbase")


def test_non_boolean_fill_flag_raises(tmp_path):
    # The contract is {'book': bool, 'trades': bool}. A non-bool value (e.g. the string "false", or
    # an int) is truthy/falsey by accident and would silently mis-route a required day — it must raise.
    bad = _calendar_dict()
    bad["coinbase_fill_days"] = {"2024-08-06": {"book": True, "trades": "false"}}
    cal = tc.load_usable_calendar(_write_calendar(tmp_path, bad))
    with pytest.raises(ValueError, match="coinbase_fill_days"):
        tc.calendar_state(cal, "2024-08-06", "coinbase")


# --------------------------------------------------------------------------- 12. GB gate (pure)
def test_estimate_trades_gb_scales_with_days_and_venues():
    venues = ["binance_perp", "binance_spot", "coinbase"]
    one = tc.estimate_trades_gb(venues, ["2025-06-01"])
    assert one == sum(tc.TRADES_GB_PER_DAY[v] for v in venues)
    assert tc.estimate_trades_gb(venues, ["2025-06-01", "2024-08-05"]) == 2 * one
    assert tc.estimate_trades_gb([], ["2025-06-01"]) == 0.0


def test_quota_decision_matches_the_run_coinbase_quality_map_pattern():
    over_auto = tc.quota_decision(est_gb=50.0, used_gb=0.0, allow_broad=False)
    assert over_auto["ok"] is False and over_auto["reason"] == "exceeds_auto_cap"
    over_head = tc.quota_decision(est_gb=295.0, used_gb=20.0, allow_broad=True)
    assert over_head["ok"] is False and over_head["reason"] == "quota_headroom"
    ok = tc.quota_decision(est_gb=1.3, used_gb=42.0, allow_broad=False)
    assert ok["ok"] is True and ok["reason"] == "ok"


# ------------------------------------------- blocking-gate regression guards (§1/§8 fail invariants)
# Each of these drives a BLOCKING reason code that lands in gate.blocking_failures. The clean
# synthetic frame never trips them, so without these a dropped/relaxed fail branch would pass silently.
def test_required_day_missing_and_empty_partition_block():
    # §1: a missing/empty required (non-fill) partition must resolve to the single blocking status
    # `fail`, so it can never escape gate.blocking_failures.
    miss = tc.validate_trade_frame(None, "coinbase", "2025-06-01")       # default route = required
    assert miss["status"] == tc.FAIL and tc.MISSING_PARTITION in miss["reason_codes"]
    empty = tc.validate_trade_frame(_clean_full_day().iloc[0:0], "coinbase", "2025-06-01")
    assert empty["status"] == tc.FAIL and tc.EMPTY_PARTITION in empty["reason_codes"]
    gate = tc.build_report([miss], meta={"generated_utc": "2026-07-02T00:00:00+00:00"})["summary"]["gate"]
    assert gate["lake_required_pass"] is False
    assert any(f["day"] == "2025-06-01" and f["venue"] == "coinbase"
               for f in gate["blocking_failures"])


def test_negative_lag_blocks_but_positive_lag_is_clean():
    df = _clean_full_day()
    df["received_time"] = df["origin_time"] - pd.Timedelta(seconds=1)   # received BEFORE origin
    res = tc.validate_trade_frame(df, "coinbase", "2025-06-01")
    assert res["status"] == tc.FAIL and tc.LAG_NEGATIVE in res["reason_codes"]
    assert res["metrics"]["recv_origin_lag_neg_frac"] > 0.99
    clean = tc.validate_trade_frame(_clean_full_day(), "coinbase", "2025-06-01")
    assert clean["metrics"]["recv_origin_lag_neg_frac"] == 0.0
    assert tc.LAG_NEGATIVE not in clean["reason_codes"]


def test_low_row_count_blocks_but_min_rows_hard_boundary_is_not_flagged():
    low = tc.validate_trade_frame(_trades_df(n=500, full_day=True), "coinbase", "2025-06-01")
    assert low["status"] == tc.FAIL and tc.ROW_COUNT_IMPLAUSIBLY_LOW in low["reason_codes"]
    assert low["metrics"]["row_count"] == 500
    # exactly min_rows_hard is inclusive-usable (strict `<`) → the hard-floor code must be absent
    boundary = tc.validate_trade_frame(_trades_df(n=tc.THRESHOLDS.min_rows_hard, full_day=True),
                                       "coinbase", "2025-06-01")
    assert tc.ROW_COUNT_IMPLAUSIBLY_LOW not in boundary["reason_codes"]


def test_nonpositive_notional_blocks_and_clean_notional_metric():
    clean = tc.notional_checks(_clean_full_day())
    assert clean["notional_sum"] > 0.0
    assert clean["notional_max_trade"] == pytest.approx(60000.0 * 0.01)
    neg = _clean_full_day()
    neg["quantity"] = -0.01                                            # Σ price×quantity < 0
    res = tc.validate_trade_frame(neg, "coinbase", "2025-06-01")
    assert res["status"] == tc.FAIL and tc.NOTIONAL_NONPOSITIVE in res["reason_codes"]


def test_missing_side_column_is_surfaced():
    # §2: `side` ∈ {buy,sell} is a required column; a normalized frame missing it is a loader/
    # normalizer schema failure that must be surfaced (aggressor/CVD features depend on it), not
    # silently pass as clean. Warn (not fail): `side` does not feed the notional bar clock (§8).
    df = _clean_full_day().drop(columns=["side"])
    res = tc.validate_trade_frame(df, "coinbase", "2025-06-01")
    assert res["status"] == tc.WARN and tc.SIDE_COLUMN_MISSING in res["reason_codes"]
    assert res["metrics"]["side_available"] is False
    ok = tc.validate_trade_frame(_clean_full_day(), "coinbase", "2025-06-01")
    assert ok["metrics"]["side_available"] is True and tc.SIDE_COLUMN_MISSING not in ok["reason_codes"]


def test_unexpected_side_value_warns():
    df = _clean_full_day()
    df.loc[0, "side"] = "unknown"
    res = tc.validate_trade_frame(df, "coinbase", "2025-06-01")
    assert res["status"] == tc.WARN and tc.SIDE_VALUE_UNEXPECTED in res["reason_codes"]
    assert "unknown" in res["metrics"]["side_values"]
    clean = tc.validate_trade_frame(_clean_full_day(), "coinbase", "2025-06-01")
    assert tc.SIDE_VALUE_UNEXPECTED not in clean["reason_codes"]


def test_within_band_single_tick_spike_blocks_via_max_abs_ret_alone():
    # The other price_spike trigger: a single-tick jump INSIDE the robust median×10 band (so
    # price_out_of_band_count == 0) still blocks on price_max_abs_ret — the corrupt print the band
    # cannot see. Isolates the max-abs-ret arm of the OR (the spike fixture trips both).
    df = _clean_full_day()
    df.loc[df.index[len(df) // 2], "price"] = 100000.0                 # 0.667 tick jump, in-band
    m = tc.price_checks(df)
    assert m["price_out_of_band_count"] == 0
    assert m["price_max_abs_ret"] > tc.THRESHOLDS.price_spike_warn
    res = tc.validate_trade_frame(df, "coinbase", "2025-06-01")
    assert res["status"] == tc.FAIL and tc.PRICE_SPIKE in res["reason_codes"]


def test_report_coerces_nonfinite_metrics_to_null(tmp_path):
    # The strict-JSON contract (§4/§6): a non-finite float metric must serialize as `null`, never
    # "Infinity"/"NaN" (allow_nan=False would otherwise raise). A bad-price frame yields an inf
    # price_max_abs_ret (div-by-zero at the 0.0 print) — the coercion path the clean fixture never hits.
    bad = tc.validate_trade_frame(_trades_df(n=2400, full_day=True, bad_price=True),
                                  "coinbase", "2025-06-01")
    assert not math.isfinite(bad["metrics"]["price_max_abs_ret"])       # inf pre-serialization
    out = tmp_path / "nonfinite.json"
    tc.write_report(tc.build_report([bad], meta={"generated_utc": "2026-07-02T00:00:00+00:00"}),
                    str(out))
    txt = out.read_text()
    assert "NaN" not in txt and "Infinity" not in txt
    assert json.loads(txt)["days"][0]["metrics"]["price_max_abs_ret"] is None   # inf → null
