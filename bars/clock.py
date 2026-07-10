"""Causal Coinbase dollar-notional bar clock (E0.3, plan §A/§C.2; issue #61).

T1 scope only: the trailing/as-of-only threshold schedule + warmup, the hybrid
time-cap clock with the monotone decision watermark, and backlog-tie coalescing.
Snapshots/features/labels/costs/orchestration are T2-T10.

Timing discipline (load-bearing, Codex P1/#13 + deep-review #2):

- Trades accumulate in (origin_time, seq) order — Coinbase trades are NOT stored in
  origin order (data.md §5b), so `bars_for_day` sorts defensively and `push` fails
  closed on out-of-order input.
- The decision time is a CUMULATIVE, NON-DECREASING watermark
  `t_event(N) = max(t_event(N-1), max(received_time) over bar N's members, cap_fire(N))`.
  Per-bar max(received_time) alone is not monotone: a delayed trade in bar N can
  arrive after bar N+1's members, which would let a later bar claim a decision time
  at which its membership was unknowable.
- A cap-fired bar is never available before its cap fire time — with few or zero
  trades the `cap_fire` term dominates, so `t_event >= cap_fire` always holds.
- The clamp can tie several backlog bars to one t_event; `coalesce_decision_bars`
  keeps only the LAST-closing bar per decision instant so downstream tasks cannot
  score one instant as multiple modeling opportunities (T9 additionally dedupes on
  `(t_event, horizon)` before write).
"""
from __future__ import annotations

import datetime as _dt
import math
from typing import Iterable, Iterator, NamedTuple

from bars.events import MIN_ABSOLUTE_NS, ClockTrade, clock_order_key


def _parse_day(day: str) -> _dt.date:
    try:
        return _dt.date.fromisoformat(day)
    except ValueError as e:
        raise ValueError(f"day must be an ISO YYYY-MM-DD date, got {day!r}") from e


class ThresholdConfig(NamedTuple):
    """Schedule parameters. All are manifest parameters (plan §H `bar_clock`);
    the values used pre-calibration are SEEDS (open question Q4), so none default."""
    target_bars_per_day: int
    window_days: int            # trailing calendar window (plan §A: 7-30d)
    warmup_days: int            # min qualifying prior days before the trailing mean is used
    seed_threshold: float       # fixed dollar-notional threshold during warmup
    min_covered_fraction: float = 0.0  # prior days below this are excluded from the mean


class DayThreshold(NamedTuple):
    day: str
    threshold: float  # dollar notional per bar in force for `day`
    is_warmup: bool   # seed threshold in use; bars are flagged for exclusion downstream


class ThresholdSchedule:
    """Per-day threshold from PRIOR days' completed volume only (strictly < d).

    Using day d's own completed volume would leak future volume into d's bar
    boundaries — a sampling look-ahead (plan §A, Codex P2b). Causality is enforced
    here by date comparison, not by recording order: `threshold_for(d)` ignores any
    recorded day >= d, so a caller cannot corrupt an earlier threshold by recording
    history late. Each prior day's raw completed notional is normalized by its
    covered fraction so a gappy/filled day does not skew every later threshold
    (Codex #12); days under `min_covered_fraction` are excluded from the mean and do
    not count toward warmup."""

    def __init__(self, config: ThresholdConfig) -> None:
        c = config
        if c.target_bars_per_day < 1:
            raise ValueError(f"target_bars_per_day must be >= 1, got {c.target_bars_per_day}")
        if c.warmup_days < 1:
            raise ValueError(f"warmup_days must be >= 1, got {c.warmup_days}")
        if c.window_days < c.warmup_days:
            raise ValueError(
                f"window_days ({c.window_days}) < warmup_days ({c.warmup_days}) "
                "would make warmup permanent"
            )
        if not (math.isfinite(c.seed_threshold) and c.seed_threshold > 0.0):
            raise ValueError(f"seed_threshold must be finite and > 0, got {c.seed_threshold}")
        if not (0.0 <= c.min_covered_fraction <= 1.0):
            raise ValueError(
                f"min_covered_fraction must be in [0, 1], got {c.min_covered_fraction}"
            )
        self.config = c
        self._history: dict[_dt.date, tuple[float, float]] = {}  # date -> (notional, coverage)

    def record_day(self, day: str, completed_notional: float,
                   covered_fraction: float = 1.0) -> None:
        """Record a day's RAW completed dollar notional after the day is processed.
        Duplicate recording fails closed — it would silently double-weight the day."""
        d = _parse_day(day)
        if d in self._history:
            raise ValueError(f"completed volume for {day} already recorded")
        if not (math.isfinite(completed_notional) and completed_notional >= 0.0):
            raise ValueError(f"completed_notional must be finite and >= 0, "
                             f"got {completed_notional}")
        if not (0.0 < covered_fraction <= 1.0):
            raise ValueError(f"covered_fraction must be in (0, 1], got {covered_fraction}")
        self._history[d] = (completed_notional, covered_fraction)

    def threshold_for(self, day: str) -> DayThreshold:
        d = _parse_day(day)
        lo = d - _dt.timedelta(days=self.config.window_days)
        # sorted by date so the float mean never depends on record_day call order
        # (window <= ~30 entries; sum() compensation alone is an interpreter detail)
        normalized = [
            notional / coverage
            for prior, (notional, coverage) in sorted(self._history.items())
            if lo <= prior < d and coverage >= self.config.min_covered_fraction
        ]
        if len(normalized) < self.config.warmup_days:
            return DayThreshold(day, self.config.seed_threshold, True)
        mean_volume = sum(normalized) / len(normalized)
        return DayThreshold(day, mean_volume / self.config.target_bars_per_day, False)


# Bar close reasons. `emitted_by_time_cap` is True ONLY for a fired time cap; a
# day-end close is a partition truncation artifact (stitch_policy's cross-midnight
# follow-up), kept distinct so T9 can mask it rather than treat it as a quiet bar.
CLOSE_THRESHOLD = "threshold"
CLOSE_TIME_CAP = "time_cap"
CLOSE_DAY_END = "day_end"


class Bar(NamedTuple):
    index: int                # position in the emission sequence (chains across days)
    interval_start_ns: int    # origin-axis start of the accumulation interval
    close_reason: str         # CLOSE_THRESHOLD | CLOSE_TIME_CAP | CLOSE_DAY_END
    cap_fire_ns: int | None   # forced-close instant (cap deadline / day end); None for threshold
    t_event: int              # monotone decision watermark (see module docstring)
    threshold: float          # dollar threshold in force for the bar's day
    is_warmup: bool           # day used the seed threshold; excluded downstream
    notional: float           # accumulated dollar notional over members
    members: tuple[ClockTrade, ...]  # the bar's trades in (origin_time, seq) order

    @property
    def emitted_by_time_cap(self) -> bool:
        return self.close_reason == CLOSE_TIME_CAP

    @property
    def trade_count(self) -> int:
        return len(self.members)

    @property
    def close_ns(self) -> int:
        """Origin-axis close: the crossing trade's origin for a threshold bar, else
        the forced-close instant (feeds the E0.3 time-per-bar diagnostics)."""
        return self.cap_fire_ns if self.cap_fire_ns is not None else self.members[-1].origin_time


class BarClock:
    """Streaming single-day dollar-bar state machine (bounded memory: only the open
    bar's members are held). Emit on cumulative notional >= threshold OR when the
    time cap fires, whichever first (spec §5.1-5.2 hybrid).

    Interval semantics (deterministic, origin axis): the accumulation interval is
    half-open `[interval_start, interval_start + time_cap_ns)` — a trade with origin
    exactly at the deadline belongs to the NEXT interval (the cap fires first). After
    a cap close the next interval starts at the fire time (contiguous grid, so a dead
    stretch becomes many small quiet bars, never one long one); after a threshold
    close it re-anchors at the crossing trade's origin. `finish` force-closes the
    day-end residual only when it holds trades — an empty partial interval was never
    a decided quiet interval, so it is not emitted.

    Timestamps are int ns on whatever absolute axis the caller pinned (the
    `bars.events` adapter guarantees absolute UTC for vendor data); this machine only
    requires day_start <= origin < day_end."""

    def __init__(self, *, threshold: float, time_cap_ns: int, day_start_ns: int,
                 day_end_ns: int, is_warmup: bool = False,
                 initial_watermark_ns: int = 0, start_index: int = 0) -> None:
        if not (math.isfinite(threshold) and threshold > 0.0):
            raise ValueError(f"threshold must be finite and > 0, got {threshold}")
        if time_cap_ns <= 0:
            raise ValueError(f"time_cap_ns must be > 0, got {time_cap_ns}")
        if day_end_ns <= day_start_ns:
            raise ValueError(f"day_end_ns ({day_end_ns}) must exceed "
                             f"day_start_ns ({day_start_ns})")
        if initial_watermark_ns < 0:
            raise ValueError(f"initial_watermark_ns must be >= 0, got {initial_watermark_ns}")
        if start_index < 0:
            raise ValueError(f"start_index must be >= 0, got {start_index}")
        self._threshold = threshold
        self._cap_ns = int(time_cap_ns)
        self._day_start = int(day_start_ns)
        self._day_end = int(day_end_ns)
        self._is_warmup = is_warmup
        self._watermark = int(initial_watermark_ns)
        self._index = int(start_index)
        self._interval_start = int(day_start_ns)
        self._members: list[ClockTrade] = []
        self._notional = 0.0
        self._last_key: tuple[int, int] | None = None
        self._finished = False

    def _close(self, reason: str, fire_ns: int | None) -> Bar:
        # THE monotone watermark: max(prev t_event, max received over members, fire).
        candidates = [self._watermark]
        if self._members:
            candidates.append(max(m.received_time for m in self._members))
        if fire_ns is not None:
            candidates.append(fire_ns)
        self._watermark = max(candidates)
        bar = Bar(index=self._index, interval_start_ns=self._interval_start,
                  close_reason=reason, cap_fire_ns=fire_ns, t_event=self._watermark,
                  threshold=self._threshold, is_warmup=self._is_warmup,
                  notional=self._notional, members=tuple(self._members))
        self._index += 1
        self._members = []
        self._notional = 0.0
        return bar

    def push(self, t: ClockTrade) -> list[Bar]:
        """Feed the next trade in (origin_time, seq) order; return bars closed by it
        (elapsed empty cap intervals first, then a possible threshold close)."""
        if self._finished:
            raise RuntimeError("BarClock.finish() was already called")
        if not (self._day_start <= t.origin_time < self._day_end):
            raise ValueError(
                f"trade origin {t.origin_time} outside the day partition "
                f"[{self._day_start}, {self._day_end}) — day routing is the caller's job"
            )
        key = clock_order_key(t)
        if self._last_key is not None and key <= self._last_key:
            raise ValueError(
                f"trade key {key} is not in strictly-increasing (origin_time, seq) "
                f"order after {self._last_key} — sort (and dedupe) before pushing"
            )
        self._last_key = key
        notional = t.price * t.amount
        if not (math.isfinite(notional) and notional > 0.0):
            raise ValueError(f"non-finite or non-positive trade notional {notional!r}")
        emitted = []
        while t.origin_time >= self._interval_start + self._cap_ns:
            fire = self._interval_start + self._cap_ns
            emitted.append(self._close(CLOSE_TIME_CAP, fire))
            self._interval_start = fire
        self._members.append(t)
        self._notional += notional
        if self._notional >= self._threshold:
            emitted.append(self._close(CLOSE_THRESHOLD, None))
            self._interval_start = t.origin_time
        return emitted

    def finish(self) -> list[Bar]:
        """Close out the day: emit every fully-elapsed cap interval, then the
        residual as a day-end truncation IF it holds trades."""
        if self._finished:
            raise RuntimeError("BarClock.finish() was already called")
        self._finished = True
        emitted = []
        while self._interval_start + self._cap_ns <= self._day_end:
            fire = self._interval_start + self._cap_ns
            emitted.append(self._close(CLOSE_TIME_CAP, fire))
            self._interval_start = fire
        if self._members:
            emitted.append(self._close(CLOSE_DAY_END, self._day_end))
        return emitted


def coalesce_decision_bars(bars: Iterable[Bar]) -> Iterator[Bar]:
    """Collapse backlog ties: yield exactly one bar per distinct t_event — the
    LAST-closing bar, the most-informed state at that instant (deep-review #2).

    Streaming and O(1): holds only the current tie candidate, so it can wrap a
    multi-day bar stream (a carry-in watermark can tie across a day boundary).
    Fails closed on a decreasing t_event — that would mean the input is not a
    single watermark-ordered stream."""
    pending: Bar | None = None
    for bar in bars:
        if pending is not None and bar.t_event < pending.t_event:
            raise ValueError(
                f"bar {bar.index} t_event {bar.t_event} < prior {pending.t_event}; "
                "input must be a non-decreasing t_event stream"
            )
        if pending is not None and bar.t_event > pending.t_event:
            yield pending
        pending = bar
    if pending is not None:
        yield pending


_DAY_NS = 86_400 * 10**9


def _day_bounds_ns(day: str) -> tuple[int, int]:
    d = _parse_day(day)
    open_ns = int(_dt.datetime(d.year, d.month, d.day,
                               tzinfo=_dt.timezone.utc).timestamp()) * 10**9
    return open_ns, open_ns + _DAY_NS


def bars_for_day(trades: Iterable[ClockTrade], *, day: str, schedule: ThresholdSchedule,
                 time_cap_ns: int, initial_watermark_ns: int = 0,
                 start_index: int = 0) -> Iterator[Bar]:
    """One day-partitioned clock run: resolve the day's threshold from the trailing
    schedule (prior days only — the threshold is fixed BEFORE any of this day's
    trades are seen), sort the day's trades defensively into (origin_time, seq)
    order, and stream bars out.

    Sorting materializes ONE day of trades — bounded and permitted (AGENTS.md
    day-partitioned rule; the trade stream is orders of magnitude smaller than the
    book stream `merge_sorted` is forbidden for). Chaining across days is the
    caller's job: pass the previous day's last t_event as `initial_watermark_ns` and
    its next bar index as `start_index`, and call
    `schedule.record_day(day, notional, coverage)` only AFTER consuming the day.

    The output is PRE-coalesce: a delayed backlog can tie several bars to one
    t_event (also across a chained day boundary), so the consumer must wrap the
    concatenated multi-day stream in `coalesce_decision_bars` before treating bars
    as modeling opportunities (T9 then dedupes on `(t_event, horizon)` at write)."""
    day_threshold = schedule.threshold_for(day)
    day_start_ns, day_end_ns = _day_bounds_ns(day)
    ordered = sorted(trades, key=clock_order_key)
    for a, b in zip(ordered, ordered[1:]):
        if clock_order_key(a) == clock_order_key(b):
            raise ValueError(f"duplicate (origin_time, seq) key {clock_order_key(a)} — "
                             "the total accumulation order would be ambiguous")
    for t in ordered:
        # BarClock is axis-agnostic, so this absolute-axis entry point owns the
        # received-time floor: an unconverted time-of-day receipt slipping past the
        # bars.events adapter would silently become a t_event decades in the past
        # (origin is already forced absolute by the day-bound check in push)
        if t.received_time < MIN_ABSOLUTE_NS:
            raise ValueError(
                f"received_time {t.received_time} at (origin_time, seq)="
                f"{clock_order_key(t)} is not an absolute UTC epoch timestamp"
            )
    clk = BarClock(threshold=day_threshold.threshold, time_cap_ns=time_cap_ns,
                   day_start_ns=day_start_ns, day_end_ns=day_end_ns,
                   is_warmup=day_threshold.is_warmup,
                   initial_watermark_ns=initial_watermark_ns, start_index=start_index)
    for t in ordered:
        yield from clk.push(t)
    yield from clk.finish()
