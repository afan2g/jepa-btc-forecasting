"""Dual-cut target-book snapshots + staleness gate (plan §B/§C.2; issue #74, T2).

T2 scope only: for every bar decision `t_event` the producer reads the SAME target
book at two different cuts and keeps the roles apart (Codex #1/#2/#8):

- **Observable feature/cost read** — only book events with `received_time <=
  t_event` (the per-event observability gate, §C.2), folded in canonical
  `(origin_time, seq)` order. `target_read_ts` is the origin time of the last such
  event; the read feeds features and `half_spread_bps` (T3/T7). `sample_topk_as_of`
  cuts on the ORIGIN axis, so it must never be pointed at the raw stream for this
  read — the received gate comes first, then the origin-order fold.
- **True label-anchor read** — the plain origin cut at `t_event` (every event with
  `origin_time <= t_event`), NOT observability-gated: the label is offline ground
  truth and `P0` is never read at `target_read_ts` (that would fold the already
  realized `[target_read_ts, t_event]` drift into the label — a common-mode leak).

Both reads reuse the existing `recon.orderbook.OrderBook` fold and its mid /
microprice formulas. The API keeps the roles hard to confuse: two distinct result
types with disjoint timestamp field names (`target_read_ts` vs `label_cut_ts`).

Timing discipline (load-bearing):

- Events fold in `(origin_time, seq)` order; equal keys fold in ORIGINAL input
  order (the stream position is the final tie-break), so rebuilds are deterministic.
- Apply-before-read is inclusive at equal timestamps on both axes: an event with
  `origin_time == t_event` enters the label cut, one with `received_time == t_event`
  passes the observability gate.
- A stale observable book (`t_event - target_read_ts > staleness_cap_ns`) and a
  missing / one-sided / invalid / crossed book fail closed as per-bar
  `SnapshotRejection` rows with stable reasons — timestamp presence is not gap
  absence (Codex #8). Contract violations (malformed events, out-of-order input,
  a non-monotone `t_event` stream, receipt before origin) raise instead: they mean
  the pipeline feeding this sampler is broken, not that one row is unusable.

Streaming/day-partitioned: one pass over the origin-ordered event stream, one pass
over the non-decreasing `t_event` stream. Memory is bounded by book depth, the set
of price levels touched in the day, and the not-yet-observable straggler buffer —
never the full day of events. Materializing helpers (`recon.reconstruct`) remain
small fixture oracles only. Source-neutral: no venue, vendor, or source mode is
named here (#67 owns the source-mode expansion); day routing and cross-day book /
straggler carry are the T9 orchestrator's job, so the first bars of a day fail
closed as `missing_book` until T9 seeds the day boundary explicitly.
"""
from __future__ import annotations

import heapq
import math
from typing import Iterable, Iterator, NamedTuple

from recon.events import Delta
from recon.orderbook import OrderBook

_SIDES = ("bid", "ask")


class BookDelta(NamedTuple):
    """Received-time-bearing L2 book event (the book-channel mirror of
    `bars.events.ClockTrade`): origin orders the fold, received gates observability."""
    origin_time: int    # ns since epoch UTC (exchange time; ordering axis)
    received_time: int  # ns since epoch UTC (capture time; observability axis)
    seq: int            # deterministic tie-break within equal origin_time
    side: str           # "bid" | "ask"
    price: float
    size: float         # absolute size at this level; 0.0 => remove the level


def book_order_key(e: BookDelta) -> tuple[int, int]:
    """Canonical fold order: (origin_time, seq). received_time never orders —
    it gates observability only (§C.2). Mirrors `bars.events.clock_order_key`."""
    return (e.origin_time, e.seq)


class ObservableBookRead(NamedTuple):
    """The feature/cost role: the box-observable book at `t_event` (§B/§C.2)."""
    target_read_ts: int  # origin time of the LAST observable target-book event
    mid: float
    microprice: float


class LabelBookRead(NamedTuple):
    """The label-anchor role: the TRUE book at the plain origin cut `t_event`."""
    label_cut_ts: int    # == t_event; the origin-axis cut the read reflects
    mid: float
    microprice: float


class BarBookReads(NamedTuple):
    """Both reads for one bar decision. The nested types keep the roles apart."""
    t_event: int
    observable: ObservableBookRead
    label: LabelBookRead


# Stable, testable per-bar rejection reasons (issue #74 fail-closed criteria).
REJECT_MISSING = "missing_book"
REJECT_ONE_SIDED = "one_sided_book"
REJECT_INVALID = "invalid_book"
REJECT_CROSSED = "crossed_book"
REJECT_STALE = "stale_book"

# Which read failed. Checked observable-first, then label; first failure wins.
ROLE_OBSERVABLE = "observable"
ROLE_LABEL = "label"


class SnapshotRejection(NamedTuple):
    """A dropped bar row: `reason` is one of the REJECT_* constants (stable API),
    `detail` is human context only."""
    t_event: int
    role: str
    reason: str
    detail: str


def validate_book_top(book: OrderBook) -> tuple[str, str] | None:
    """Fail-closed shape check of a book's top: (reason, detail) or None if usable.

    Order (deterministic, documented): missing -> one_sided -> invalid -> crossed.
    Crossed uses the repo convention `best_bid >= best_ask` (recon/parity.py) — a
    locked book is not a tradable state either."""
    bb, ba = book.best_bid(), book.best_ask()
    if bb is None and ba is None:
        return REJECT_MISSING, "book has no levels on either side"
    if bb is None or ba is None:
        side = "ask" if ba is None else "bid"
        return REJECT_ONE_SIDED, f"book has no {side} levels"
    bs, as_ = book.bids[bb], book.asks[ba]
    for name, v in (("best_bid", bb), ("best_ask", ba),
                    ("best_bid_size", bs), ("best_ask_size", as_)):
        if not (math.isfinite(v) and v > 0.0):
            return REJECT_INVALID, f"{name} {v!r} is not finite and positive"
    if bb >= ba:
        return REJECT_CROSSED, f"best_bid {bb} >= best_ask {ba}"
    return None


def _validate_event(e: BookDelta) -> None:
    if e.side not in _SIDES:
        raise ValueError(f"unrecognized book side {e.side!r}; expected bid/ask "
                         "(vendor forms must be normalized at ingestion)")
    if not (math.isfinite(e.price) and e.price > 0.0):
        raise ValueError(f"non-positive or non-finite price {e.price!r} at "
                         f"(origin_time, seq)={book_order_key(e)}")
    if not (math.isfinite(e.size) and e.size >= 0.0):
        raise ValueError(f"negative or non-finite size {e.size!r} at "
                         f"(origin_time, seq)={book_order_key(e)}")
    if e.received_time < e.origin_time:
        # the two-axis contract: capture cannot precede the exchange event. A
        # violating source would let an origin>t_event event pass the received
        # gate, making target_read_ts exceed t_event and staleness negative.
        raise ValueError(
            f"received_time {e.received_time} < origin_time {e.origin_time} at "
            f"(origin_time, seq)={book_order_key(e)} — the source's timestamp "
            "contract is broken (certify/normalize at ingestion)"
        )


def dual_book_reads(events: Iterable[BookDelta], t_events: Iterable[int], *,
                    staleness_cap_ns: int) -> Iterator[BarBookReads | SnapshotRejection]:
    """Stream the dual target-book reads for a day: one `BarBookReads` or
    `SnapshotRejection` per entry of `t_events`, in order.

    Contract (fail-closed):
      * `events` is iterable in non-decreasing `(origin_time, seq)` order (equal
        keys allowed — they fold in input order). Only the prefix with
        `origin_time <= max(t_events)` is consumed/validated (streaming).
      * `t_events` is STRICTLY increasing — one decision per instant; a duplicate
        means the caller skipped `bars.clock.coalesce_decision_bars`.
      * `staleness_cap_ns >= 0`; a bar whose observable book is older than the cap
        (`t_event - target_read_ts > staleness_cap_ns`) is rejected as stale.

    The observable book is maintained incrementally: a straggler that becomes
    observable after later-origin events were already folded applies to a price
    level only when its `(origin_time, seq, input position)` fold key exceeds the
    level's last-applied key — exactly the fold of the received-gated set in origin
    order, without re-replaying the day per bar (bounded memory)."""
    if staleness_cap_ns < 0:
        raise ValueError(f"staleness_cap_ns must be >= 0, got {staleness_cap_ns}")

    ev_iter = iter(events)
    label_book = OrderBook()
    obs_book = OrderBook()
    # (side, price) -> last fold key applied to that level in the OBSERVABLE fold.
    # Kept for removals too (a tombstone): an older straggler must not resurrect a
    # level a newer removal emptied.
    level_key: dict[tuple[str, float], tuple[int, int, int]] = {}
    # min-heap of not-yet-observable events keyed by (received_time, position)
    pending: list[tuple[int, int, BookDelta]] = []
    last_event_key: tuple[int, int] | None = None
    position = 0            # input ordinal: the final, unique fold tie-break
    obs_last_key: tuple[int, int, int] | None = None  # max fold key seen observable
    prev_t: int | None = None
    lookahead: BookDelta | None = next(ev_iter, None)

    for t_event in t_events:
        t_event = int(t_event)
        if prev_t is not None and t_event <= prev_t:
            raise ValueError(
                f"t_event {t_event} does not increase past {prev_t}; the decision "
                "stream must be strictly increasing (coalesce backlog ties first)"
            )
        prev_t = t_event

        # 1) advance the origin cursor: fold every event with origin <= t_event
        #    into the TRUE label book, and queue it for observability promotion.
        while lookahead is not None and lookahead.origin_time <= t_event:
            e = lookahead
            _validate_event(e)
            key = book_order_key(e)
            if last_event_key is not None and key < last_event_key:
                raise ValueError(
                    f"book event key {key} is out of (origin_time, seq) order "
                    f"after {last_event_key} — sort the stream before sampling"
                )
            last_event_key = key
            label_book.apply(Delta(e.origin_time, e.seq, e.side, e.price, e.size))
            heapq.heappush(pending, (e.received_time, position, e))
            position += 1
            lookahead = next(ev_iter, None)

        # 2) promote events that became observable: received_time <= t_event.
        while pending and pending[0][0] <= t_event:
            _, pos, e = heapq.heappop(pending)
            key = (e.origin_time, e.seq, pos)
            lvl = (e.side, e.price)
            if level_key.get(lvl, (-1, -1, -1)) < key:
                level_key[lvl] = key
                obs_book.apply(Delta(e.origin_time, e.seq, e.side, e.price, e.size))
            if obs_last_key is None or key > obs_last_key:
                obs_last_key = key

        # 3) fail-closed validation, observable role first, then staleness, then
        #    the label role; first failure wins (deterministic).
        bad = validate_book_top(obs_book)
        if bad is not None:
            yield SnapshotRejection(t_event, ROLE_OBSERVABLE, bad[0], bad[1])
            continue
        target_read_ts = obs_last_key[0]
        age = t_event - target_read_ts
        if age > staleness_cap_ns:
            yield SnapshotRejection(
                t_event, ROLE_OBSERVABLE, REJECT_STALE,
                f"observable book age {age} ns exceeds cap {staleness_cap_ns} ns")
            continue
        bad = validate_book_top(label_book)
        if bad is not None:
            yield SnapshotRejection(t_event, ROLE_LABEL, bad[0], bad[1])
            continue

        yield BarBookReads(
            t_event=t_event,
            observable=ObservableBookRead(target_read_ts=target_read_ts,
                                          mid=obs_book.mid(),
                                          microprice=obs_book.microprice()),
            label=LabelBookRead(label_cut_ts=t_event,
                                mid=label_book.mid(),
                                microprice=label_book.microprice()),
        )
