"""Causal stationarized single-venue per-bar features (plan §H/§C.2/E1.2; issue #78, T3).

T3 scope only: the pinned G0 feature registry computed from (a) the bar's exact
origin-ordered member trades (T1 `bars.clock.Bar`) and (b) the box-observable
target-book reads (T2 `bars.snapshot.ObservableBookRead`, now carrying the top-K
ladder). Labels, costs, manifests, cross-venue features, and orchestration are
T4-T10 and do NOT live here.

Input discipline (load-bearing):

- **Trade-flow features fold exactly `bar.members`** (Codex #3) — never the
  received-gated superset. Members are observable by construction (`t_event >=`
  every member's `received_time`, the T1 monotone watermark); this module
  re-checks that invariant and fails loudly if a pipeline hands it a bar whose
  membership was not decidable at its own `t_event`.
- **Book features consume only observable state**: the current bar's
  `ObservableBookRead` and (for OFI) the previous bar's — both are received-gated
  at their own `t_event`s upstream (T2), so every book input satisfies
  `received_time <= t_event`. A `LabelBookRead` is rejected by type: the true
  label-anchor book is `P0`'s ground truth and must never leak into features
  (plan §B/§C.2 role separation).
- **`t_feature_start` is the true feature-window start**: the oldest origin-axis
  instant the retained features actually consume =
  `min(bar.interval_start_ns, prev_read.target_read_ts)`. Book reads count as
  point-state inputs at their read timestamps (their fold history is book STATE,
  not feature look-back — the same convention that puts `P0` at `t_event`); the
  bar's accumulation interval is consumed from `interval_start_ns` (elapsed /
  intensity measure the whole interval, including its trade-free stretches). The
  prior read can be much older than the current bar during rejection runs or
  book-quiet stretches — the emitted value reports it honestly; T6/T8 own the
  robust look-back cap and `max_lookback_ns` sizing (plan §F Codex #13/#B).

Stationarization (plan §H "every stationarizer as-of <= t_event"): no full-window
statistics exist here at all. Every feature is scale-free (bounded ratios, bps
offsets, tick units, log returns) or normalized by the bar's trailing dollar
threshold — which T1 fixes from PRIOR days only, so the only "normalizer state"
is as-of by construction. The whole vector is invariant under the price/size/tick
rescaling (price x lambda, size / lambda, tick x lambda); raw price levels never
enter a feature column.

Pinned formulas (hand-fixture-tested; sign conventions: buy aggressor = +,
bid-side liquidity gain = +):

- `ofi_integrated`  multi-level/integrated OFI (E1.2): level-indexed
  order-flow-imbalance between the previous and current observable top-K ladders
  (per level: price improved -> +cur size; unchanged -> size delta; retreated ->
  -prev size; ask side mirrored), summed over levels 0..K-1 as
  sum(e_bid - e_ask), then x `mid / threshold` (bar-threshold units). Plan §C.2
  lists `ofi_integrated` in its member-trade enumeration; the pinned E1.2/§H
  definition is the *multi-level/integrated* book OFI, which is exactly why the
  observable read exposes top-K state — the implementation keeps §C.2's actual
  constraints (no received-gated superset, all inputs observable by t_event) via
  the two bar-boundary reads. Needs a prior read: the first bar of a chain fails
  closed (`no_prior_read`) unless T9 seeds `initial_read` across the boundary.
- `microprice_dev`  1e4 * (microprice - mid) / mid  [bps]
- `queue_imb`       (best_bid_size - best_ask_size) / (sum)  in (-1, 1)
- `spread_tick`     (best_ask - best_bid) / tick_size  [ticks]
- `cvd`             sum(+- price*amount) / threshold  (signed notional, threshold units)
- `depth_imbalance` (sum bid sizes - sum ask sizes) / (total), top-K  in (-1, 1)
- `book_slope`      (top-K depth * mid / threshold) / ladder span in bps, where
                    span = 1e4 * (ask_price[K-1] - bid_price[K-1]) / mid
- `vwap_minus_mid`  1e4 * (member VWAP - mid) / mid  [bps]
- `trade_count`     len(members)
- `signed_vol`      sum(+- amount) / sum(amount)  in [-1, 1] (volume-weighted sign)
- `aggressor_imb`   (n_buy - n_sell) / n  in [-1, 1] (count-weighted sign)
- `largest_print`   max(price*amount) / threshold
- `event_intensity` log1p(trade_count * 1e9 / max(elapsed_ns, 1))  [log trades/s;
                    the 1 ns floor keeps a same-instant burst finite]
- `rv_intrabar`     sum of squared member-trade log returns x 1e8  [bps^2]
- `mae_intrabar`    max adverse excursion of the member-trade price path against
                    the bar's open->close direction, in bps (>= 0)
- `elapsed_ns`      bar.close_ns - bar.interval_start_ns (bounded by the time cap)
- `tod_sin/tod_cos` sin/cos(2*pi * UTC time-of-day fraction of t_event)

Pinned edge policies (plan §E Codex #4 — legit emissions stay finite, real
problems fail closed, nothing is silently zero-filled *as if observed*):

- Zero-/one-trade bars: trade-flow features are DEFINED as 0 on an empty member
  set (the §E pinned policy — "no aggression in this interval" is an observation,
  not a gap); book-shape features still come from the observable read.
- Insufficient ladder depth (< top_k levels on either side of either read):
  `FeatureRejection(insufficient_depth)` — the row is dropped, never padded.
- No prior observable read: `FeatureRejection(no_prior_read)`.
- Every rejection still advances the read chain (state is observation, not
  emission), so one thin/first read costs exactly the affected bars.
- Contract violations (label-role input, malformed members/ladders, out-of-order
  bars or reads, membership not observable by t_event) RAISE: they mean the
  T1/T2 pipeline feeding this builder is broken, not that one row is unusable.

T9 wiring obligations (documented here because T3 has no orchestrator):
run `dual_book_reads` with `top_k >= FeatureConfig.top_k` (a smaller sampler
depth is indistinguishable from a thin book and rejects every row); pair each
coalesced `Bar` with the `BarBookReads` of the SAME `t_event` and pass
`.observable` only; skip `SnapshotRejection` bars entirely (the builder chain
tolerates the gap); carry the last observable read across day boundaries via
`initial_read` (else each day's first bar is `no_prior_read`); drop `is_warmup`
bars downstream as T1 already flags them.

Streaming and bounded: the builder holds one prior read and one prior t_event —
never a bar history, never the day's events.
"""
from __future__ import annotations

import math
from typing import NamedTuple

from bars.clock import Bar
from bars.events import ClockTrade, clock_order_key
from bars.snapshot import LabelBookRead, ObservableBookRead

_DAY_NS = 86_400 * 10**9

# The pinned G0 registry (plan §H) — explicit ordered names; T8 writes these into
# `feature_cols` verbatim. Order is the manifest order.
FEATURE_COLS = (
    "ofi_integrated", "microprice_dev", "queue_imb", "spread_tick", "cvd",
    "depth_imbalance", "book_slope", "vwap_minus_mid", "trade_count",
    "signed_vol", "aggressor_imb", "largest_print", "event_intensity",
    "rv_intrabar", "mae_intrabar", "elapsed_ns", "tod_sin", "tod_cos",
)

# Stable, testable per-bar rejection reasons (mirrors bars.snapshot REJECT_*).
REJECT_INSUFFICIENT_DEPTH = "insufficient_depth"
REJECT_NO_PRIOR_READ = "no_prior_read"


class FeatureConfig(NamedTuple):
    """Feature parameters. Both are manifest parameters (plan §H `bar_clock` /
    `sources`); #67 binds the per-instrument values, so none default."""
    top_k: int        # required ladder depth per side (spec §6: K=10-20)
    tick_size: float  # exchange price increment for `spread_tick`


class FeatureRow(NamedTuple):
    """One bar's emitted feature vector. `t_event` is the join key (post-coalesce
    unique); `t_feature_start` feeds the §E timing columns and §F look-back."""
    t_event: int
    t_feature_start: int
    ofi_integrated: float
    microprice_dev: float
    queue_imb: float
    spread_tick: float
    cvd: float
    depth_imbalance: float
    book_slope: float
    vwap_minus_mid: float
    trade_count: float
    signed_vol: float
    aggressor_imb: float
    largest_print: float
    event_intensity: float
    rv_intrabar: float
    mae_intrabar: float
    elapsed_ns: float
    tod_sin: float
    tod_cos: float


assert FeatureRow._fields == ("t_event", "t_feature_start") + FEATURE_COLS


class FeatureRejection(NamedTuple):
    """A dropped bar row: `reason` is one of the REJECT_* constants (stable API),
    `detail` is human context only."""
    t_event: int
    reason: str
    detail: str


def _validate_config(c: FeatureConfig) -> None:
    if c.top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {c.top_k}")
    if not (math.isfinite(c.tick_size) and c.tick_size > 0.0):
        raise ValueError(f"tick_size must be finite and > 0, got {c.tick_size}")


def _validate_read(read: ObservableBookRead, label: str) -> None:
    """Fail loudly on a read this module must never consume or that T2 could not
    have produced (role misuse / hand-built corruption), not on thin-but-honest
    depth — that is the data-driven `insufficient_depth` rejection."""
    if isinstance(read, LabelBookRead):
        raise ValueError(
            f"{label} is a LabelBookRead — the true label-anchor book is P0's "
            "ground truth and must never feed features (plan §B/§C.2)"
        )
    if not isinstance(read, ObservableBookRead):
        raise ValueError(f"{label} must be an ObservableBookRead, got {type(read).__name__}")
    for side, prices, sizes, ordered in (
        ("bid", read.bid_prices, read.bid_sizes, lambda a, b: a > b),
        ("ask", read.ask_prices, read.ask_sizes, lambda a, b: a < b),
    ):
        if len(prices) != len(sizes):
            raise ValueError(
                f"{label} {side} depth tuples are ragged "
                f"({len(prices)} prices vs {len(sizes)} sizes)"
            )
        for v in prices:
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"{label} {side} depth price {v!r} is not finite and > 0")
        for v in sizes:
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"{label} {side} depth size {v!r} is not finite and > 0")
        for a, b in zip(prices, prices[1:]):
            if not ordered(a, b):
                raise ValueError(
                    f"{label} {side} depth prices {prices} are not strictly "
                    f"{'descending' if side == 'bid' else 'ascending'} from the best"
                )
    if read.bid_prices and read.ask_prices and read.bid_prices[0] >= read.ask_prices[0]:
        raise ValueError(
            f"{label} depth ladder is crossed: best bid {read.bid_prices[0]} >= "
            f"best ask {read.ask_prices[0]}"
        )
    for name in ("mid", "microprice", "best_bid", "best_ask",
                 "best_bid_size", "best_ask_size"):
        v = getattr(read, name)
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError(f"{label} {name} {v!r} is not finite and > 0")
    if read.bid_prices and (read.bid_prices[0] != read.best_bid
                            or read.bid_sizes[0] != read.best_bid_size):
        raise ValueError(f"{label} depth level 0 disagrees with the best bid fields")
    if read.ask_prices and (read.ask_prices[0] != read.best_ask
                            or read.ask_sizes[0] != read.best_ask_size):
        raise ValueError(f"{label} depth level 0 disagrees with the best ask fields")


_SIDES = ("buy", "sell")


def _validate_bar(bar: Bar) -> None:
    if not (math.isfinite(bar.threshold) and bar.threshold > 0.0):
        raise ValueError(f"bar threshold must be finite and > 0, got {bar.threshold}")
    if bar.close_ns < bar.interval_start_ns:
        raise ValueError(
            f"bar close {bar.close_ns} precedes interval start {bar.interval_start_ns}"
        )
    last_key = None
    for m in bar.members:
        if m.side not in _SIDES:
            raise ValueError(f"unrecognized member side {m.side!r}; expected buy/sell")
        if not (math.isfinite(m.price) and m.price > 0.0):
            raise ValueError(f"non-positive or non-finite member price {m.price!r}")
        if not (math.isfinite(m.amount) and m.amount > 0.0):
            raise ValueError(f"non-positive or non-finite member amount {m.amount!r}")
        if m.received_time < m.origin_time:
            raise ValueError(
                f"member received_time {m.received_time} < origin_time "
                f"{m.origin_time} — the source's timestamp contract is broken"
            )
        if m.received_time > bar.t_event:
            raise ValueError(
                f"member received_time {m.received_time} > bar t_event "
                f"{bar.t_event} — membership was not decidable at t_event; the "
                "T1 monotone watermark cannot produce this"
            )
        if m.origin_time < bar.interval_start_ns:
            raise ValueError(
                f"member origin {m.origin_time} precedes the bar interval start "
                f"{bar.interval_start_ns}"
            )
        key = clock_order_key(m)
        if last_key is not None and key <= last_key:
            raise ValueError(
                f"member key {key} is not in strictly-increasing (origin_time, "
                f"seq) order after {last_key} — the clock's canonical member "
                "order was not preserved"
            )
        last_key = key


def _level_flow(p_prev: float, s_prev: float, p_cur: float, s_cur: float,
                improved: bool, retreated: bool) -> float:
    if improved:
        return s_cur
    if retreated:
        return -s_prev
    return s_cur - s_prev


def _side_ofi(prev_p, prev_s, cur_p, cur_s, *, is_bid: bool, k: int) -> float:
    total = 0.0
    for l in range(k):
        pp, pc = prev_p[l], cur_p[l]
        improved = pc > pp if is_bid else pc < pp
        retreated = pc < pp if is_bid else pc > pp
        total += _level_flow(pp, prev_s[l], pc, cur_s[l], improved, retreated)
    return total


class BarFeatureBuilder:
    """Streaming per-bar feature builder (bounded state: one prior read + one
    prior t_event). Feed coalesced bars in strictly-increasing `t_event` order,
    each paired with ITS observable read; see the module docstring for the T9
    wiring obligations and the pinned edge policies."""

    def __init__(self, config: FeatureConfig, *,
                 initial_read: ObservableBookRead | None = None) -> None:
        _validate_config(config)
        if initial_read is not None:
            _validate_read(initial_read, "initial_read")
        self._cfg = config
        self._prev_read = initial_read
        self._prev_t: int | None = None

    def build(self, bar: Bar, read: ObservableBookRead) -> FeatureRow | FeatureRejection:
        _validate_read(read, "read")
        if self._prev_t is not None and bar.t_event <= self._prev_t:
            raise ValueError(
                f"bar t_event {bar.t_event} does not increase past {self._prev_t}; "
                "coalesce backlog ties and feed bars in decision order"
            )
        if read.target_read_ts > bar.t_event:
            raise ValueError(
                f"read target_read_ts {read.target_read_ts} > bar t_event "
                f"{bar.t_event} — the read is from a later decision (mispaired)"
            )
        prev = self._prev_read
        if prev is not None and read.target_read_ts < prev.target_read_ts:
            raise ValueError(
                f"read target_read_ts {read.target_read_ts} regressed past the "
                f"prior read's {prev.target_read_ts} — reads must come from one "
                "forward pass"
            )
        _validate_bar(bar)
        # observation advances the chain even when this bar's row is rejected
        self._prev_read = read
        self._prev_t = bar.t_event

        if prev is None:
            return FeatureRejection(
                bar.t_event, REJECT_NO_PRIOR_READ,
                "no prior observable read — seed initial_read across the "
                "chain/day boundary (T9) or drop the first bar")
        k = self._cfg.top_k
        for label, r in (("prior", prev), ("current", read)):
            short = min(len(r.bid_prices), len(r.ask_prices))
            if short < k:
                return FeatureRejection(
                    bar.t_event, REJECT_INSUFFICIENT_DEPTH,
                    f"{label} observable read has {short} ladder level(s) < "
                    f"required top_k {k} (thin book or under-provisioned sampler)")

        mid = read.mid
        threshold = bar.threshold

        # --- book shape (observable read only)
        ofi_raw = (_side_ofi(prev.bid_prices, prev.bid_sizes,
                             read.bid_prices, read.bid_sizes, is_bid=True, k=k)
                   - _side_ofi(prev.ask_prices, prev.ask_sizes,
                               read.ask_prices, read.ask_sizes, is_bid=False, k=k))
        ofi_integrated = ofi_raw * mid / threshold
        microprice_dev = 1e4 * (read.microprice - mid) / mid
        queue_imb = ((read.bid_sizes[0] - read.ask_sizes[0])
                     / (read.bid_sizes[0] + read.ask_sizes[0]))
        spread_tick = (read.ask_prices[0] - read.bid_prices[0]) / self._cfg.tick_size
        bid_depth = sum(read.bid_sizes[:k])
        ask_depth = sum(read.ask_sizes[:k])
        depth_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
        span_bps = 1e4 * (read.ask_prices[k - 1] - read.bid_prices[k - 1]) / mid
        book_slope = ((bid_depth + ask_depth) * mid / threshold) / span_bps

        # --- trade flow over exactly the bar's members (zero policy: plan §E #4)
        members = bar.members
        n = len(members)
        if n:
            signed_notional = signed_amount = total_amount = total_notional = 0.0
            n_buy = 0
            largest = 0.0
            for m in members:
                sign = 1.0 if m.side == "buy" else -1.0
                notional = m.price * m.amount
                signed_notional += sign * notional
                signed_amount += sign * m.amount
                total_amount += m.amount
                total_notional += notional
                largest = max(largest, notional)
                n_buy += m.side == "buy"
            cvd = signed_notional / threshold
            signed_vol = signed_amount / total_amount
            aggressor_imb = (2 * n_buy - n) / n
            largest_print = largest / threshold
            vwap = total_notional / total_amount
            vwap_minus_mid = 1e4 * (vwap - mid) / mid
        else:
            cvd = signed_vol = aggressor_imb = largest_print = vwap_minus_mid = 0.0

        # --- intra-bar path over the member-trade price sequence
        if n >= 2:
            rv_intrabar = sum(
                (1e4 * math.log(b.price / a.price)) ** 2
                for a, b in zip(members, members[1:])
            )
            p0 = members[0].price
            direction = 1.0 if members[-1].price >= p0 else -1.0
            mae_intrabar = max(
                0.0, max(-direction * 1e4 * math.log(m.price / p0) for m in members)
            )
        else:
            rv_intrabar = mae_intrabar = 0.0

        # --- time / intensity
        elapsed = bar.close_ns - bar.interval_start_ns
        elapsed_ns = float(elapsed)
        event_intensity = math.log1p(n * 1e9 / max(elapsed, 1)) if n else 0.0
        tod = 2.0 * math.pi * ((bar.t_event % _DAY_NS) / _DAY_NS)

        row = FeatureRow(
            t_event=bar.t_event,
            t_feature_start=min(bar.interval_start_ns, prev.target_read_ts),
            ofi_integrated=ofi_integrated,
            microprice_dev=microprice_dev,
            queue_imb=queue_imb,
            spread_tick=spread_tick,
            cvd=cvd,
            depth_imbalance=depth_imbalance,
            book_slope=book_slope,
            vwap_minus_mid=vwap_minus_mid,
            trade_count=float(n),
            signed_vol=signed_vol,
            aggressor_imb=aggressor_imb,
            largest_print=largest_print,
            event_intensity=event_intensity,
            rv_intrabar=rv_intrabar,
            mae_intrabar=mae_intrabar,
            elapsed_ns=elapsed_ns,
            tod_sin=math.sin(tod),
            tod_cos=math.cos(tod),
        )
        for name, v in zip(FEATURE_COLS, row[2:]):
            if not math.isfinite(v):  # unreachable given the validations; belt&braces
                raise ValueError(f"non-finite feature {name}={v!r} at t_event {bar.t_event}")
        return row
