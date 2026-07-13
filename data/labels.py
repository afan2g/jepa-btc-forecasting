"""Causal forward-return and triple-barrier labels (E0.4; plan §B/§D/§E, T5; issue #79).

T5 scope only: turn the TRUE target-mid path and the bar decision anchors into one
deterministic label row per `(t_event, horizon)`:

- **`P0` discipline (plan #1, load-bearing).** The label base price is the true
  reconstructed target mid at `t_event` — T2's label-anchor read
  (`bars.snapshot.LabelBookRead.mid`), a plain origin cut. It is NEVER the
  observable `target_read_ts` book: that would fold the already-realized,
  feature-observable `[target_read_ts, t_event]` drift into `y_fwd_bps` (a
  common-mode target leak the E0.4 control cannot catch). Entry latency is a
  T7 `cost_bps` concern, not a label shift.
- **Horizontal barriers** at `P0 * (1 ± width_bps/1e4)`, width = `width_mult *
  max(EWMA vol, vol_floor_bps)` where the EWMA is trailing/as-of-only: it folds
  the bps returns between consecutive path points whose end timestamp is
  `<= t_event` (the return ending exactly AT `t_event` legitimately feeds it),
  decayed by `0.5 ** (dt / halflife_ns)` per observation gap. No return after
  `t_event` can influence the width. Every estimator parameter is explicit in
  `BarrierParams` and persistable via `as_dict()` for the T8 manifest.
- **Resolution** scans the causal true-mid path over `(t_event, t_event +
  horizon]` (both barriers checked per point; a touch is a hit — `>=`/`<=`
  inclusive; the horizon endpoint is inside the window). The first hit in time
  wins and `t_barrier` is that actual hit time. If no horizontal barrier fires,
  the row keeps the realized as-of return at `t_event + horizon`, `label = 0`,
  and `t_barrier = t_event + horizon` (the vertical resolution time).
- **Equal-time ties are pinned by coalescing:** several path points at one
  timestamp collapse to the LAST in input order (the book state AT that
  instant, matching T2's apply-before-read fold); intermediate same-timestamp
  states never fire a barrier. A two-sided same-point tie is structurally
  impossible because the width is validated/rejected to be `> 0`, so the up
  barrier sits strictly above the down barrier.

Emission order is deterministic: anchors in their (strictly increasing) input
order, horizons ascending by duration. Streaming/day-partitioned: one forward
pass over the path with memory bounded by the largest horizon window, never the
full day — pre-anchor history (however long the warm-up prefix) is folded into
the trailing EWMA point by point and discarded, never buffered.
"""
from __future__ import annotations

import math
import numbers
from collections import deque
from types import MappingProxyType
from typing import Iterable, Iterator, Mapping, NamedTuple

# The default physical-time ladder (plan §D). E1.1's ~20-30s τ-rung is added by
# rebuilding labels with an extended mapping, never by editing emitted rows.
DEFAULT_HORIZONS: Mapping[str, int] = MappingProxyType(
    {"2s": 2_000_000_000, "10s": 10_000_000_000, "60s": 60_000_000_000})

# Estimator identity persisted with the parameters (manifest provenance).
ESTIMATOR = "ewma_bps_time_halflife_v1"

# Stable per-row rejection reasons (the T2 SnapshotRejection pattern): these are
# expected market/warm-up states, not broken contracts, so they yield rows the
# T9 orchestrator can count instead of raising.
REJECT_INSUFFICIENT_HISTORY = "insufficient_vol_history"
REJECT_DEGENERATE_WIDTH = "degenerate_barrier_width"


class MidPoint(NamedTuple):
    """One point of the TRUE target-mid path (origin-time axis, T2 label fold)."""
    ts: int      # ns since epoch UTC
    mid: float


class LabelAnchor(NamedTuple):
    """One bar decision: `p0` is T2's true label-anchor mid at `t_event`."""
    t_event: int
    p0: float


class LabelRow(NamedTuple):
    """One resolved `(t_event, horizon)` label row."""
    t_event: int
    horizon: str        # horizon tag, e.g. "2s" (duration lives in the params)
    y_fwd_bps: float    # forward return over [t_event, t_barrier], bps of P0
    label: int          # +1 up-barrier, -1 down-barrier, 0 vertical/time
    t_barrier: int      # ACTUAL first-hit time, or t_event + horizon if vertical
    p0: float           # the true mid at t_event the return is anchored on
    width_bps: float    # as-of horizontal barrier half-width used for this row


class LabelRejection(NamedTuple):
    """A dropped `(t_event, horizon)` row: `reason` is one of the REJECT_*
    constants (stable API), `detail` is human context only."""
    t_event: int
    horizon: str
    reason: str
    detail: str


class BarrierParams(NamedTuple):
    """Every horizontal-barrier estimator parameter (persist via `as_dict()`)."""
    halflife_ns: int            # EWMA time-decay half-life (per-gap 0.5**(dt/hl))
    min_returns: int            # trailing returns required before a width exists
    width_mult: float           # width = width_mult * max(vol_bps, vol_floor_bps)
    vol_floor_bps: float = 0.0  # lower bound on the EWMA vol entering the width
    horizons: Mapping[str, int] = DEFAULT_HORIZONS

    def as_dict(self) -> dict:
        """Manifest-ready copy of every parameter plus the estimator identity."""
        validate_barrier_params(self)
        return {
            "estimator": ESTIMATOR,
            "horizons": {tag: int(ns) for tag, ns in self.horizons.items()},
            "halflife_ns": int(self.halflife_ns),
            "min_returns": int(self.min_returns),
            "width_mult": float(self.width_mult),
            "vol_floor_bps": float(self.vol_floor_bps),
        }


def _int_param(name: str, v) -> int:
    if isinstance(v, bool) or not isinstance(v, numbers.Integral):
        raise ValueError(f"{name} must be an integer number of nanoseconds/counts; "
                         f"got {v!r}")
    return int(v)


def _float_param(name: str, v) -> float:
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise ValueError(f"{name} must be a real number; got {v!r}")
    f = float(v)
    if not math.isfinite(f):
        raise ValueError(f"{name} must be finite; got {f!r}")
    return f


def validate_barrier_params(params: BarrierParams) -> None:
    """Fail closed on any inexplicit/degenerate estimator configuration."""
    if not params.horizons:
        raise ValueError("horizons must be a non-empty {tag: duration_ns} mapping")
    seen_ns: dict[int, str] = {}
    for tag, ns in params.horizons.items():
        if not isinstance(tag, str) or not tag:
            raise ValueError(f"horizon tags must be non-empty strings; got {tag!r}")
        nsv = _int_param(f"horizons[{tag!r}]", ns)
        if nsv <= 0:
            raise ValueError(f"horizons[{tag!r}] must be > 0 ns; got {nsv}")
        if nsv in seen_ns:
            raise ValueError(f"horizons {seen_ns[nsv]!r} and {tag!r} share duration "
                             f"{nsv} ns; one physical horizon must have one tag")
        seen_ns[nsv] = tag
    if _int_param("halflife_ns", params.halflife_ns) <= 0:
        raise ValueError(f"halflife_ns must be > 0; got {params.halflife_ns}")
    if _int_param("min_returns", params.min_returns) < 1:
        raise ValueError(f"min_returns must be >= 1; got {params.min_returns}")
    if _float_param("width_mult", params.width_mult) <= 0.0:
        raise ValueError(f"width_mult must be > 0; got {params.width_mult}")
    if _float_param("vol_floor_bps", params.vol_floor_bps) < 0.0:
        raise ValueError(f"vol_floor_bps must be >= 0; got {params.vol_floor_bps}")


def anchor_from_bar_reads(reads) -> LabelAnchor:
    """Adapter from T2's `bars.snapshot.BarBookReads` (duck-typed so `data/`
    does not import `bars/`): P0 is the TRUE label-anchor mid,
    `reads.label.mid`. It deliberately never reads `reads.observable` — an
    observable/`target_read_ts` P0 folds the realized, feature-observable
    pre-decision drift into the label (plan #1), and the labeler's own
    P0-vs-path consistency check fails closed on such a miswired anchor."""
    return LabelAnchor(t_event=int(reads.t_event), p0=float(reads.label.mid))


def _point_ts(raw) -> int:
    """Peek ONLY the timestamp of a raw path point. A lookahead row past every
    needed window must be positionable without validating its mid (deep-review
    P2: a malformed or next-partition value just past the last supported
    window must not kill the supported labels — it is never consumed)."""
    try:
        ts_raw, _mid = raw
    except (TypeError, ValueError):
        raise ValueError(
            f"path points must be (ts, mid) pairs; got {raw!r}") from None
    return _int_param("path point ts", ts_raw)


def _validated_point(raw, prev_ts) -> tuple[int, float]:
    try:
        ts_raw, mid_raw = raw
    except (TypeError, ValueError):
        raise ValueError(
            f"path points must be (ts, mid) pairs; got {raw!r}") from None
    ts = _int_param("path point ts", ts_raw)
    mid = _float_param("path point mid", mid_raw)
    if mid <= 0.0:
        raise ValueError(f"path point mid must be positive; got {mid!r} at ts {ts}")
    if prev_ts is not None and ts < prev_ts:
        raise ValueError(f"path timestamps are out of order: {ts} after {prev_ts}; "
                         "the true-mid path must be non-decreasing")
    return ts, mid


def triple_barrier_labels(path: Iterable, anchors: Iterable, *,
                          params: BarrierParams,
                          coverage_end_ns) -> Iterator[LabelRow | LabelRejection]:
    """Stream one `LabelRow` or `LabelRejection` per `(anchor, horizon)`, in
    anchor order then ascending horizon duration (module docstring pins the
    semantics). Validates the configuration eagerly, then iterates lazily.

    Contract (fail-closed):
      * `path` is the TRUE target-mid path — `(ts, mid)` pairs in non-decreasing
        `ts` order (equal `ts` allowed: the last point at an instant is the
        state, earlier ones are coalesced away). Mids must be finite and
        positive. Only the prefix needed by the anchors is consumed/validated
        (streaming; the tail past the last anchor's window stays untouched,
        and a lookahead row past every needed window is only position-peeked —
        its value is never validated).
      * `anchors` are `(t_event, p0)` with strictly increasing `t_event` — a
        duplicate decision key means the caller skipped backlog coalescing —
        and `p0` equal to the true path mid at `t_event` (T2's label-anchor
        read; the observable read fails this check by construction whenever
        they differ).
      * `coverage_end_ns` declares how far the path is COMPLETE. Any
        `(t_event, horizon)` window ending past it is refused before the
        forward path is consumed — the T9 partition prefilter must drop or
        re-batch boundary rows per horizon; T5 never opens the next partition.
    """
    validate_barrier_params(params)
    coverage_end = _int_param("coverage_end_ns", coverage_end_ns)
    ladder = sorted(((tag, int(ns)) for tag, ns in params.horizons.items()),
                    key=lambda kv: kv[1])
    return _label_iter(iter(path), iter(anchors), ladder, params, coverage_end)


_STREAM_UNOPENED = object()   # lookahead sentinel: the path was never touched


def _label_iter(path_iter, anchor_iter, ladder, params, coverage_end):
    max_tag, max_h = ladder[-1]
    halflife = params.halflife_ns
    pts: deque[tuple[int, float]] = deque()   # coalesced points kept for scanning
    # The lookahead is NOT primed here: the first anchor's boundary-refusal
    # check must run before the stream is touched at all (deep-review P2 — a
    # lazy chained partition iterator must not be opened for a refused window).
    nxt = _STREAM_UNOPENED                    # lookahead raw path point
    prev_raw_ts: int | None = None
    # trailing EWMA state over consecutive coalesced-point returns (bps)
    ewma_s = 0.0
    ewma_w = 0.0
    n_returns = 0
    last: tuple[int, float] | None = None     # last EWMA-consumed (ts, mid)
    prev_t: int | None = None

    def fold(ts: int, mid: float) -> None:
        """Consume one coalesced point into the trailing EWMA; `last` becomes
        the as-of state. Callers only ever fold in strictly increasing ts
        order (buffered window points first, then freshly pulled ones)."""
        nonlocal ewma_s, ewma_w, n_returns, last
        if last is not None:
            r = (mid - last[1]) / last[1] * 1e4
            d = 0.5 ** ((ts - last[0]) / halflife)
            ewma_s = d * ewma_s + r * r
            ewma_w = d * ewma_w + 1.0
            n_returns += 1
        last = (ts, mid)

    for anchor in anchor_iter:
        try:
            t_raw, p_raw = anchor
        except (TypeError, ValueError):
            raise ValueError(
                f"anchors must be (t_event, p0) pairs; got {anchor!r}") from None
        t_event = _int_param("anchor t_event", t_raw)
        p0 = _float_param("anchor p0", p_raw)
        if p0 <= 0.0:
            raise ValueError(f"anchor p0 must be positive; got {p0!r} at "
                             f"t_event {t_event}")
        if prev_t is not None and t_event <= prev_t:
            raise ValueError(
                f"anchor t_event {t_event} does not increase past {prev_t}; "
                "decision keys must be strictly increasing — one row per "
                "(t_event, horizon), coalesce backlog ties upstream")
        prev_t = t_event

        # end-of-partition refusal BEFORE the forward path is consumed: the
        # caller's per-horizon prefilter owns boundary drops (plan §E), so a
        # window overhanging the declared coverage is a broken pipeline here.
        if t_event + max_h > coverage_end:
            raise ValueError(
                f"insufficient future support: t_event {t_event} + horizon "
                f"{max_tag!r} ends at {t_event + max_h} > coverage_end_ns "
                f"{coverage_end}; the partition prefilter must drop or re-batch "
                "boundary rows per horizon (never open the next partition here)")

        # fold buffered window points that this anchor has moved past (they
        # were pulled for an earlier anchor's scan and now end <= t_event)
        while pts and pts[0][0] <= t_event:
            fold(*pts.popleft())

        # pull every coalesced path point with ts <= t_event + max horizon.
        # Pre-anchor points fold into the EWMA immediately and are DISCARDED
        # (Codex P2: a long warm-up/filtered prefix must never be enqueued
        # wholesale); only in-window points are buffered for the barrier scans.
        if nxt is _STREAM_UNOPENED:           # first supported anchor opens it
            nxt = next(path_iter, None)
        bound = t_event + max_h
        while nxt is not None:
            if _point_ts(nxt) > bound:        # peek position only: an out-of-
                break                         # window value is never validated
            ts, mid = _validated_point(nxt, prev_raw_ts)
            prev_raw_ts = ts
            nxt = next(path_iter, None)
            while nxt is not None:                       # same-ts: last wins
                if _point_ts(nxt) != ts:
                    break
                _, mid = _validated_point(nxt, prev_raw_ts)
                nxt = next(path_iter, None)
            if ts <= t_event:
                fold(ts, mid)
            else:
                pts.append((ts, mid))

        # P0 discipline: the anchor must BE the true path state at t_event
        if last is None:
            raise ValueError(
                f"anchor t_event {t_event} has no path point at or before it; "
                "the true-mid path does not support P0")
        if last[1] != p0:
            raise ValueError(
                f"anchor p0 {p0!r} does not equal the true path mid {last[1]!r} "
                f"at t_event {t_event}; P0 must be T2's label-anchor read (the "
                "true book at t_event), never the observable target_read_ts book")

        if n_returns < params.min_returns:
            for tag, _h in ladder:
                yield LabelRejection(
                    t_event, tag, REJECT_INSUFFICIENT_HISTORY,
                    f"only {n_returns} trailing return(s) at t_event; "
                    f"min_returns={params.min_returns}")
            continue
        vol_bps = math.sqrt(ewma_s / ewma_w)
        width = params.width_mult * max(vol_bps, params.vol_floor_bps)
        if width <= 0.0:
            for tag, _h in ladder:
                yield LabelRejection(
                    t_event, tag, REJECT_DEGENERATE_WIDTH,
                    f"EWMA vol {vol_bps!r} bps with vol_floor_bps "
                    f"{params.vol_floor_bps!r} gives a non-positive barrier width")
            continue

        for tag, h in ladder:
            end = t_event + h
            asof_mid = p0
            hit = None
            for ts, mid in pts:                # invariant: every ts > t_event
                if ts > end:
                    break
                ret = (mid - p0) / p0 * 1e4
                if ret >= width:
                    hit = (ts, ret, 1)
                    break
                if ret <= -width:
                    hit = (ts, ret, -1)
                    break
                asof_mid = mid
            if hit is not None:
                yield LabelRow(t_event, tag, hit[1], hit[2], hit[0], p0, width)
            else:
                y = (asof_mid - p0) / p0 * 1e4
                yield LabelRow(t_event, tag, y, 0, end, p0, width)
