"""Causal dollar-notional bar clock (issue #61 / plan §A, §C.2, §J tier-1).

Covers: threshold crossing, hybrid time cap + emitted_by_time_cap, zero/low-trade cap
bars, the MONOTONE decision watermark
`t_event(N) = max(t_event(N-1), max(received_time) over members, cap_fire(N))`
(Codex P1/#13 — non-decreasing across bars, a cap bar never available before its cap
fire), deterministic origin-then-seq accumulation (Coinbase trades are NOT
origin-sorted, data.md §5b), backlog-tie coalescing (deep-review #2: at most one
modeling opportunity per decision instant, last-closing bar wins), and the
day-partitioned glue over the trailing threshold schedule."""
import itertools
import random

import pandas as pd
import pytest

from bars.clock import (
    CLOSE_DAY_END,
    CLOSE_THRESHOLD,
    CLOSE_TIME_CAP,
    Bar,
    BarClock,
    ThresholdConfig,
    ThresholdSchedule,
    bars_for_day,
    coalesce_decision_bars,
)
from bars.events import ClockTrade

_SEQ = itertools.count()


def trade(origin, received=None, *, notional=100.0, seq=None, side="buy"):
    """Notional-first fixture: price 100, amount = notional/100."""
    return ClockTrade(
        origin_time=origin,
        received_time=origin if received is None else received,
        seq=next(_SEQ) if seq is None else seq,
        side=side,
        price=100.0,
        amount=notional / 100.0,
    )


def clock(*, threshold=1_000.0, cap=10_000, day_start=0, day_end=100_000,
          watermark=0, start_index=0, is_warmup=False):
    return BarClock(threshold=threshold, time_cap_ns=cap, day_start_ns=day_start,
                    day_end_ns=day_end, is_warmup=is_warmup,
                    initial_watermark_ns=watermark, start_index=start_index)


def run(clk, trades):
    bars = []
    for t in trades:
        bars.extend(clk.push(t))
    bars.extend(clk.finish())
    return bars


# ---------------------------------------------------------------- threshold crossing

def test_bar_closes_on_the_threshold_crossing_trade():
    trades = [trade(1_000 * i, notional=300.0) for i in range(1, 6)]
    bars = run(clock(), trades)
    burst = bars[0]
    assert burst.close_reason == CLOSE_THRESHOLD
    assert burst.cap_fire_ns is None
    assert burst.emitted_by_time_cap is False
    assert burst.trade_count == 4  # cum 300,600,900,1200 -> crossing at the 4th trade
    assert burst.notional == pytest.approx(1_200.0)
    assert burst.members == tuple(trades[:4])
    # the 5th trade opens the next accumulation interval at the crossing origin
    assert bars[1].members[0] == trades[4]


def test_threshold_bar_t_event_is_max_member_received_time():
    # the EARLIER member is received LATER than the crossing trade, so an
    # implementation using the trigger's own receipt (8_000 vs 9_000) would fail
    trades = [trade(1_000, 9_000, notional=500.0), trade(2_000, 8_000, notional=500.0)]
    bars = run(clock(), trades)
    assert bars[0].close_reason == CLOSE_THRESHOLD
    assert bars[0].members[-1].origin_time == 2_000  # crossing trade
    assert bars[0].t_event == 9_000  # max received over members, not the trigger's


# ------------------------------------------------------------------ hybrid time cap

def test_time_cap_fires_when_threshold_not_crossed():
    bars = run(clock(), [trade(1_000, notional=100.0), trade(2_000, notional=100.0),
                         trade(15_000, notional=100.0)])
    lull = bars[0]
    assert lull.close_reason == CLOSE_TIME_CAP
    assert lull.emitted_by_time_cap is True
    assert lull.cap_fire_ns == 10_000
    assert lull.trade_count == 2


def test_zero_trade_intervals_emit_empty_cap_bars():
    bars = run(clock(), [trade(25_000, notional=2_000.0)])
    assert [(b.close_reason, b.cap_fire_ns, b.trade_count) for b in bars[:2]] == [
        (CLOSE_TIME_CAP, 10_000, 0), (CLOSE_TIME_CAP, 20_000, 0)]
    assert bars[2].close_reason == CLOSE_THRESHOLD  # the trade itself crosses


def test_trade_exactly_at_cap_boundary_belongs_to_the_next_interval():
    # intervals are half-open [start, start + cap): origin == cap fire time is beyond it
    bars = run(clock(), [trade(10_000, notional=2_000.0)])
    assert bars[0] == Bar(index=0, interval_start_ns=0, close_reason=CLOSE_TIME_CAP,
                          cap_fire_ns=10_000, t_event=10_000, threshold=1_000.0,
                          is_warmup=False, notional=0.0, members=())
    assert bars[1].members[0].origin_time == 10_000


def test_cap_fired_bar_is_never_available_before_its_cap_fire_time():
    # members received long before the cap fires: availability is still the fire time
    bars = run(clock(), [trade(1_000, 1_100, notional=100.0)])
    quiet = bars[0]
    assert quiet.close_reason == CLOSE_TIME_CAP
    assert quiet.t_event == quiet.cap_fire_ns == 10_000


def test_cap_grid_restarts_at_a_threshold_close():
    bars = run(clock(), [trade(4_000, notional=1_500.0), trade(5_000, notional=10.0),
                         trade(15_000, notional=10.0)])
    assert bars[0].close_reason == CLOSE_THRESHOLD
    # next interval anchors at the crossing origin 4_000, so its cap fires at 14_000
    assert bars[1].close_reason == CLOSE_TIME_CAP
    assert bars[1].cap_fire_ns == 14_000
    assert [m.origin_time for m in bars[1].members] == [5_000]


# ------------------------------------------------------------------- day-end flush

def test_day_end_residual_with_members_is_emitted_as_day_end_bar():
    clk = clock(day_end=95_000)
    bars = run(clk, [trade(92_000, notional=100.0)])
    tail = bars[-1]
    assert tail.close_reason == CLOSE_DAY_END
    assert tail.emitted_by_time_cap is False
    assert tail.cap_fire_ns == 95_000
    assert tail.t_event >= 95_000  # not decidable before the partition closes
    assert tail.trade_count == 1


def test_empty_day_yields_only_full_cap_intervals_and_no_residual():
    bars = run(clock(day_end=95_000), [])
    assert len(bars) == 9  # [0,10k) ... [80k,90k); the partial [90k,95k) is empty
    assert {b.close_reason for b in bars} == {CLOSE_TIME_CAP}
    assert all(b.trade_count == 0 for b in bars)
    assert [b.cap_fire_ns for b in bars] == [10_000 * i for i in range(1, 10)]


def test_push_after_finish_fails_closed():
    clk = clock()
    clk.finish()
    with pytest.raises(RuntimeError):
        clk.push(trade(1_000))
    with pytest.raises(RuntimeError):
        clk.finish()


# ------------------------------------------------------- monotone decision watermark

def test_delayed_receipt_clamps_later_bars_t_event_non_decreasing():
    # bar 1 holds a trade received at 50_000; bar 2's members are all received by
    # 22_000 — without the clamp t_event would go backwards (Codex P1/#13)
    trades = [trade(1_000, 50_000, notional=1_000.0),
              trade(21_000, 21_500, notional=1_000.0)]
    bars = run(clock(cap=100_000, day_end=100_000), trades)
    assert [b.close_reason for b in bars] == [CLOSE_THRESHOLD, CLOSE_THRESHOLD]
    assert bars[0].t_event == 50_000
    assert bars[1].t_event == 50_000  # clamped to the prior watermark, not 21_500
    assert all(b1.t_event <= b2.t_event for b1, b2 in zip(bars, bars[1:]))


def test_t_event_equals_the_cumulative_watermark_formula_for_every_bar():
    # mixed threshold / cap / delayed / empty / day-end bars, checked against an
    # independent recomputation of max(prev, max received over members, cap_fire)
    trades = [
        trade(1_000, 45_000, notional=600.0),   # delayed receipt
        trade(2_000, 2_100, notional=600.0),    # threshold close at 2_000
        trade(12_000, 12_500, notional=100.0),  # lands after the first cap fires
        trade(33_000, 90_000, notional=100.0),  # delayed again, cap-closed later
        trade(41_000, 41_200, notional=2_000.0),
    ]
    clk = clock(day_start=0, day_end=50_000, cap=10_000, watermark=7)
    bars = run(clk, trades)
    watermark = 7
    for b in bars:
        candidates = [watermark]
        if b.members:
            candidates.append(max(m.received_time for m in b.members))
        if b.cap_fire_ns is not None:
            candidates.append(b.cap_fire_ns)
        watermark = max(candidates)
        assert b.t_event == watermark, b
    assert all(b1.t_event <= b2.t_event for b1, b2 in zip(bars, bars[1:]))


def test_initial_watermark_carries_across_day_boundaries():
    clk = clock(watermark=123_456_789)
    bars = run(clk, [trade(500, 600, notional=2_000.0)])
    assert bars[0].t_event == 123_456_789  # prior day's late receipt still clamps


# ----------------------------------------------------- deterministic ordering guards

def test_push_rejects_out_of_order_and_duplicate_keys():
    clk = clock()
    clk.push(trade(5_000, seq=10, notional=10.0))
    with pytest.raises(ValueError, match="order"):
        clk.push(trade(4_000, seq=11, notional=10.0))
    with pytest.raises(ValueError, match="order"):
        clk.push(trade(5_000, seq=10, notional=10.0))  # duplicate (origin, seq)
    clk.push(trade(5_000, seq=11, notional=10.0))  # same origin, higher seq is fine


def test_push_rejects_trades_outside_the_day_partition():
    clk = clock(day_start=1_000, day_end=50_000)
    with pytest.raises(ValueError, match="day"):
        clk.push(trade(999, notional=10.0))
    with pytest.raises(ValueError, match="day"):
        clk.push(trade(50_000, notional=10.0))


def test_push_rejects_non_finite_or_non_positive_notional():
    clk = clock()
    with pytest.raises(ValueError):
        clk.push(ClockTrade(1_000, 1_000, 1, "buy", 100.0, float("nan")))
    with pytest.raises(ValueError):
        clk.push(ClockTrade(1_000, 1_000, 2, "buy", -1.0, 1.0))


def test_invalid_clock_config_fails_closed():
    with pytest.raises(ValueError):
        clock(threshold=0.0)
    with pytest.raises(ValueError):
        clock(cap=0)
    with pytest.raises(ValueError):
        clock(day_start=10, day_end=10)
    with pytest.raises(ValueError):
        clock(watermark=-1)


# ------------------------------------------------------------- backlog-tie coalesce

def test_backlog_ties_coalesce_to_the_last_closing_bar():
    # one hugely delayed receipt clamps three subsequent single-trade bars to one
    # decision instant; only the LAST-closing bar may remain a modeling opportunity
    trades = [
        trade(10_000, 500_000, notional=1_000.0, seq=1),
        trade(20_000, 21_000, notional=1_000.0, seq=2),
        trade(30_000, 31_000, notional=1_000.0, seq=3),
        trade(40_000, 600_000, notional=1_000.0, seq=4),
    ]
    clk = clock(cap=1_000_000, day_start=0, day_end=1_000_000)
    bars = run(clk, trades)
    assert [b.t_event for b in bars] == [500_000, 500_000, 500_000, 600_000]
    decided = list(coalesce_decision_bars(bars))
    assert [b.index for b in decided] == [2, 3]  # last-closing bar of the tie, then next
    assert [b.t_event for b in decided] == [500_000, 600_000]


def test_coalesce_passes_distinct_t_events_through_unchanged():
    trades = [trade(1_000 * i, notional=1_000.0) for i in range(1, 4)]
    bars = run(clock(cap=1_000_000, day_end=1_000_000), trades)
    assert list(coalesce_decision_bars(bars)) == bars


def test_coalesce_rejects_a_decreasing_t_event_stream():
    a = run(clock(cap=1_000_000, day_end=1_000_000), [trade(1_000, 5_000, notional=1_000.0)])
    b = run(clock(cap=1_000_000, day_end=1_000_000), [trade(1_000, 4_000, notional=1_000.0)])
    with pytest.raises(ValueError, match="non-decreasing"):
        list(coalesce_decision_bars([a[0], b[0]]))


def test_coalesce_is_streaming_and_lazy():
    gen = coalesce_decision_bars(iter([]))
    assert list(gen) == []


# ------------------------------------------------------------------- E0.3 shape gate

def test_median_bar_time_on_a_planted_active_regime_is_at_most_two_seconds():
    # active regime: $250 notional every 100 ms against a $1000 threshold -> a bar
    # roughly every 400 ms; the E0.3-shape PASS/FAIL is median <= 2 s (plan §J)
    ns = 10**9
    trades = [trade(i * ns // 10, notional=250.0) for i in range(1, 600)]
    clk = clock(threshold=1_000.0, cap=5 * ns, day_start=0, day_end=60 * ns)
    bars = [b for b in run(clk, trades) if b.close_reason == CLOSE_THRESHOLD]
    closes = [b.close_ns for b in bars]
    gaps = sorted(b - a for a, b in zip(closes, closes[1:]))
    assert gaps[len(gaps) // 2] <= 2 * ns


# ------------------------------------------------------------- day-partitioned glue

DAY = "2025-01-07"
DAY_OPEN = int(pd.Timestamp(DAY, tz="UTC").value)
DAY_NS = 86_400 * 10**9

CFG = ThresholdConfig(target_bars_per_day=10, window_days=7, warmup_days=2,
                      seed_threshold=1_000.0)


def _schedule_with_history():
    s = ThresholdSchedule(CFG)
    s.record_day("2025-01-05", 20_000.0)
    s.record_day("2025-01-06", 40_000.0)
    return s


def _day_trades(n=8, notional=1_000.0):
    return [trade(DAY_OPEN + i * 10**9, DAY_OPEN + i * 10**9 + 5_000_000,
                  notional=notional, seq=i) for i in range(1, n + 1)]


def test_bars_for_day_resolves_threshold_and_warmup_from_the_schedule():
    bars = list(bars_for_day(_day_trades(), day=DAY, schedule=_schedule_with_history(),
                             time_cap_ns=DAY_NS))
    assert all(b.threshold == pytest.approx(3_000.0) for b in bars)  # mean 30k / 10
    assert all(b.is_warmup is False for b in bars)
    warm = list(bars_for_day(_day_trades(), day=DAY, schedule=ThresholdSchedule(CFG),
                             time_cap_ns=DAY_NS))
    assert all(b.threshold == 1_000.0 for b in warm)
    assert all(b.is_warmup is True for b in warm)


def test_bars_for_day_is_a_streaming_generator():
    gen = bars_for_day(_day_trades(), day=DAY, schedule=_schedule_with_history(),
                       time_cap_ns=DAY_NS)
    assert next(gen).close_reason  # yields without materializing all bars


def test_bars_for_day_scrambled_input_yields_identical_bars():
    trades = _day_trades(n=20)
    expected = list(bars_for_day(trades, day=DAY, schedule=_schedule_with_history(),
                                 time_cap_ns=DAY_NS))
    for seed in (1, 2, 3):
        scrambled = trades[:]
        random.Random(seed).shuffle(scrambled)
        got = list(bars_for_day(scrambled, day=DAY, schedule=_schedule_with_history(),
                                time_cap_ns=DAY_NS))
        assert got == expected


def test_equal_origin_trades_accumulate_in_seq_order_within_a_bar():
    # same origin_time, distinct seq: the seq tie-break must be load-bearing in the
    # actual bar membership, not only in the sort key (data.md §5b: Coinbase trades
    # are unsorted and trade_id is non-monotonic — only (origin, seq) is a total order)
    shared = DAY_OPEN + 5 * 10**9
    trades = [trade(shared, notional=1_000.0, seq=3), trade(shared, notional=1_000.0, seq=1),
              trade(shared, notional=1_000.0, seq=2)]
    bars = list(bars_for_day(trades, day=DAY, schedule=_schedule_with_history(),
                             time_cap_ns=DAY_NS))
    assert [m.seq for m in bars[0].members] == [1, 2, 3]  # threshold 3000: one bar
    assert bars[0].close_reason == CLOSE_THRESHOLD


def test_bars_for_day_rejects_duplicate_order_keys():
    t = _day_trades(n=2)
    dup = [t[0], t[0]._replace(price=200.0, amount=1.0), t[1]]
    with pytest.raises(ValueError, match="duplicate"):
        list(bars_for_day(dup, day=DAY, schedule=_schedule_with_history(),
                          time_cap_ns=DAY_NS))


def test_bars_for_day_rejects_trades_outside_the_day():
    early = [trade(DAY_OPEN - 1, notional=100.0, seq=0)]
    with pytest.raises(ValueError, match="day"):
        list(bars_for_day(early, day=DAY, schedule=_schedule_with_history(),
                          time_cap_ns=DAY_NS))


def test_bars_for_day_rejects_non_absolute_received_time():
    # the adapter floors both axes, but bars_for_day is the absolute-axis entry
    # point for ANY caller: a raw 13:00 time-of-day receipt on a hand-built
    # ClockTrade would otherwise become a t_event ~55 years before the day open
    bad = [trade(DAY_OPEN + 1_000, 13 * 3600 * 10**9, notional=100.0, seq=0)]
    with pytest.raises(ValueError, match="absolute"):
        list(bars_for_day(bad, day=DAY, schedule=_schedule_with_history(),
                          time_cap_ns=DAY_NS))


def test_bars_for_day_chains_watermark_and_index_across_days():
    carry = DAY_OPEN + 10 * 10**9  # a prior-day receipt observed after this day opened
    bars = list(bars_for_day(_day_trades(n=3), day=DAY, schedule=_schedule_with_history(),
                             time_cap_ns=DAY_NS, initial_watermark_ns=carry,
                             start_index=41))
    assert bars[0].t_event == carry
    assert [b.index for b in bars] == list(range(41, 41 + len(bars)))
